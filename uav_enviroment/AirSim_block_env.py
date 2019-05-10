import cv2
import time
import math
import sys
import os
import gym
import random
import numpy as np
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import collections
import itertools as IT
from shapely.geometry import Point, asPoint, asPolygon
from shapely.geometry.polygon import Polygon
from shapely import speedups
import shapely.vectorized as sv
from shapely.ops import cascaded_union
from shapely.prepared import prep
import copy

from uav_enviroment.env_utils import *
from uav_enviroment.env_plotting import *
# import Box2D
# from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)


from uav_enviroment.UAV_Environment import UAVEnv


max_timestep = 1000
REWARD_NORMALIZATION_FACTOR = 10 * max_timestep
OUT_OF_BOUNDS_REWARD = 1.2 * (10 * max_timestep)
# TIME_NEGATIVE_REWARD = OUT_OF_BOUNDS_REWARD/4
TIME_NEGATIVE_REWARD = 0

POSITIVE_REWARD = 1.2 * (10 * max_timestep)

EMPTY_TRAJECTORY = {'x': [],  # Position.x
                    'y': [],  # Position y
                    'vel': [],  # Velocity
                    'dist': [],  # Distance
                    'reward': []  # Reward
                    }

# OBS:
# [distance_to_target/MAX_DIST , orientation_to_target, speed_norm/MAX_VEL, p_crash, theta_crash]
# ACT:
# [thrust,steering_angle]
MAX_DIST = 100
MAX_VEL = 50
ENV_OBS_LOWS = np.array([0, -1, 0, 0, -1])
ENV_OBS_HIGH = np.array([1, 1, 1, 1, 1])
ENV_ACT_LOWS = np.array([0, -1])
ENV_ACT_HIGH = np.array([10, 1])

NUM_DISCRETE_ACTIONS = 5

STATE_SIZE = 6

threshold_distance = 40
threshold_velocity = 4

N_OBSTACLES = 0  # Better keep it under 75 to speed up
PERCEPTION_DISTANCE = 60
IMAGE_WIDTH = 300
IMAGE_HEIGTH = 300


OBS_IMAGE_WIDTH = 600
OBS_IMAGE_HEIGTH = 600

STEERING_VEL = 0.1

FRAMES_PER_SECOND = 5
SCALING_FACTOR = 4
GAMMA = 0.99
RENDER_EPISODE = 5


path_to_airsimclient = '/home/daniel/Desktop/AIRSIM/AirSim-master/PythonClient/'

os.chdir(path_to_airsimclient)
from AirSimClient import *

# Helper Classes if needed

class Airsim_UAVEnv(UAVEnv):
    continuous = False  # Denotes if the action space is continuous
    dense_reward = False  # True if the reard is dense, else it will be +1 if success -1 if not
    angular_movement = False  # When True dynamics are modeled with steer and thrust, else they are modeled with dx, dy
    observation_with_image = False  # When True return an image alongside the observation
    reset_always = False  # When True change the environment origin, target and obstacles every time it resets.
    controlled_speed = False  # When True the task is to reach the target with a controlled speed, else is just reach the target

    logging_episodes = 1000

    map_size_x = 160
    map_size_y = 240
    map_min_x = - map_size_x * 0.1
    map_min_y = - map_size_y * 0.1
    map_max_x = map_size_x * 1.1
    map_max_y = map_size_y * 1.1
    threshold_dist = 15
    threshold_vel = 4

    n_obstacles = 9

    max_timestep = 1500

    def __init__(self):
        self.seed()
        self.viewer = None

        self.continuous = False  # Denotes if the action space is continuous
        dense_reward = False  # True if the reard is dense, else it will be +1 if success -1 if not
        self.angular_movement = False  # When True dynamics are modeled with steer and thrust, else they are modeled with dx, dy
        self.observation_with_image = False  # When True return an image alongside the observation
        self.reset_always = False  # When True change the environment origin, target and obstacles every time it resets.
        self.controlled_speed = False  # When True the task is to reach the target with a controlled speed, else is just reach the target
        # Shapely Speedups
        # if speedups.available:
        speedups.enable()

        # TODO things that shouldn't be here
        self.render_episode = False
        self.num_envs = 1

        # # move this outside the env (check if they serve the resetalways flag)
        # self.latest_results = collections.deque(maxlen=150)

        self.done = [False]
        self.reward_range = (-OUT_OF_BOUNDS_REWARD, POSITIVE_REWARD)
        self.spec = None

        # Counters
        self.episode = 0
        self.total_timestep = 0
        self.episode_success = True
        self.n_done = 0
        self.oo_time = 0
        self.oob = 0
        self.crashes = 0

        # Rendering variables
        self.trajectory = copy.deepcopy(EMPTY_TRAJECTORY)
        self.RENDER_FLAG = False

        # Simulation parameters
        self.time_step = 0.1  # To obtain trajectories

        u_max = MAX_VEL / np.sqrt(2)
        v_max = MAX_VEL / np.sqrt(2)
        u_min = MAX_VEL / np.sqrt(2)
        v_min = MAX_VEL / np.sqrt(2)

        # Dynamic characteristics
        um = max([abs(u_max), abs(u_min)])
        vm = max([abs(v_max), abs(v_min)])
        self.w = np.sqrt(um ** 2 + vm ** 2)
        self.F = SCALING_FACTOR  # It was 20 in airsim i think
        self.k = self.F / self.w

        # Obstacles Variables
        self.obstacles = []
        self.prep_obstacles = []
        self.obstacle_centers = np.array([])
        self.culled = []

        # IMPLEMENTED THE IMAGE AS PARTIAL OBSERVABILITY
        perception_shape = (IMAGE_HEIGTH, IMAGE_WIDTH, 3)
        observation_shape = (OBS_IMAGE_HEIGTH, OBS_IMAGE_WIDTH, 3)
        self.field_of_view = np.zeros(perception_shape, np.uint8)

        # State definition
        self.s = {'x': [0.0],  # Position.x
                  'y': [0.0],  # Position y
                  'u': [0.0],  # Velocity x
                  'v': [0.0],  # Velocity y
                  'target_x': [0.0],
                  'target_y': [0.0],
                  'origin_x': [0.0],
                  'origin_y': [0.0]
                  }


        # Perception matrix initialization, only with cartesian(dx,dy) dynamics.
        self.perception_matrix = create_perception_matrix(dist=PERCEPTION_DISTANCE, n_radius=16, pts_p_radius=6)
        self.perception_ray_points = create_perception_distance_points(max_dist=PERCEPTION_DISTANCE, n_radius=16)
        n_discrete_actions = 4  # +dx, -dx, +dy, -dy
        env_act_lows = np.array([-1, -1])  # dx and dy
        env_act_high = np.array([1, 1])

        # Define the action space either CONTINUOUS or DISCRETE.

        self.action_space = spaces.Discrete(n_discrete_actions)

        ENV_OBS_LOWS = np.zeros(shape=self.get_cartesian_observation().shape)
        ENV_OBS_HIGH = np.ones(shape=self.get_cartesian_observation().shape)
            # ENV_OBS_HIGH[0:6] = [self.map_max_x, self.map_max_y, MAX_VEL, MAX_VEL, self.map_max_x, self.map_max_y]

        # self.observation_space = spaces.Box(low=ENV_OBS_LOWS, high=ENV_OBS_HIGH, dtype=np.float32)


        self.observation_space = spaces.Box(low=ENV_OBS_LOWS, high=ENV_OBS_HIGH, dtype=np.float32)


        self.reward_function = self._no_speed_reward()  # no speed reward


    def setup(self, *, map_size_x=map_size_x, map_size_y=map_size_y, n_obstacles=0, reset_always=True,
              max_timestep=1000, threshold_dist=40, threshold_vel=4, reward_sparsity=True):

        self.map_size_x = 160
        self.map_size_y = 240
        self.map_min_x = - self.map_size_x * 0.1
        self.map_min_y = - self.map_size_y * 0.1
        self.map_max_x = self.map_size_x * 1.1
        self.map_max_y = self.map_size_y * 1.1
        self.reset_always = False
        self.max_timestep = 1500
        self.threshold_dist = 15
        self.threshold_vel = 4

        self.n_obstacles = 9

        # True is sparse reward, False is a dense reward
        self.reward_sparsity = reward_sparsity
        self.dense_reward = not reward_sparsity

        self.get_observation = self.get_airsim_observation

        self._take_action = self._airsim_action  # Discrete cartesian

        # Obstacles of the block env
        scaled_points = \
            [[160., 0., 140., 20.],
             [20., 0., 0., 20.],
             [30., 60., 20., 70.],
             [130., 52.5, 110., 70.],
             [80., 80., 60., 100.],
             [160., 100., 120., 140.],
             [80., 140., 60., 160.],
             [30., 170., 20., 180.],
             [130., 170., 110., 190.],
             [160., 220., 140., 240.],
             [20., 220., 0., 240.]]

        for obstacle in scaled_points:
            x0, y0 = obstacle[2], obstacle[1]
            w = obstacle[0] - obstacle[2]
            h = obstacle[3] - obstacle[1]

            p1, p2, p3, p4 = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
            # print([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)])
            self.add_obstacle(asPolygon([p1, p2, p3, p4]))




        # Added these 2 lines to reduce number of polys and increase performance
        # self.obstacles = cascaded_union(self.obstacles)
        # self.prep_obstacles = [prep(polygon) for polygon in self.obstacles]
        self.prep_obstacles = self.obstacles
        self.generate_map()

        self.s['origin_x'], self.s['origin_y'] = [35.], [117.39583333]  # Scaled old [0, 0]
        self.s['target_x'], self.s['target_y'] = [35.], [117.39583333]  # Scaled old [0, 0]

        self.client = MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.move_to(35. , 117.39583333)

        self.client.hover()


    def generate_map(self):

        # Ball positions
        scaled_objectives = \
            [[42.6, 205.3],
             [13.6, 121.],
             [153.1, 186.5],
             [70.6, 12.8],
             [147.2, 59.4],
             [90.5, 158.2]]

        # TODO check if origin and target are inside []
        # self.obstacle_centers = np.array([obstacle.centroid.coords[0] for obstacle in self.obstacles])
        target = random.choice(scaled_objectives)
        if self.s['target_x'][0]:
            self.s['origin_x'], self.s['origin_y'] = self.s['target_x'], self.s['target_y']

            while target == [self.s['target_x'][0], self.s['target_y'][0]]:
                target = random.choice(scaled_objectives)

            self.s['target_x'], self.s['target_y'] = [target[0]], [target[1]]

        else:
            self.s['origin_x'], self.s['origin_y'] = [35.], [117.39583333]  # Scaled old [0, 0]
            self.s['target_x'], self.s['target_y'] = [target[0]], [target[1]]




    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        # print('Reset, episode', self.episode, ' next target:', self.target)
        self.episode += 1
        self.total_timestep = 0
        self.trajectory = copy.deepcopy(EMPTY_TRAJECTORY)

        # TODO revisar esto
        # If success get new target
        if self.episode_success:
            self.generate_map()
            self.episode_success = False

        # TODO test
        self.move_to(self.s['origin_x'][0], self.s['origin_y'][0])

        self.client.hover()
        # TODO something to keep heigth

        # # Origin and target are modified in the generate map function
        # self.s['x'] = self.s['origin_x']
        # self.s['y'] = self.s['origin_y']
        # self.s['u'] = [np.multiply(np.random.uniform(size=1), 0.1)[0] - 0.05]
        # self.s['v'] = [np.multiply(np.random.uniform(size=1), 0.1)[0] - 0.05]

        observation = self.get_observation()
        self.done = np.array([False])

        return observation

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.total_timestep += 1
        # if self.total_timestep % 20 == 0:
        #     print('TTL = ', self.total_timestep)
        # print( 'TTL = ', self.total_timestep)
        # print('action = ',action)

        # Making sure action has the right size
        action = np.reshape(action, (action.size,))

        x, y, u, v, t_x, t_y = self.get_state()

        # TODO test
        # Take the action
        x_next, y_next, u_next, v_next = self._take_action(action, x, y, u, v)

        # self.s['x'] = [x_next]
        # self.s['y'] = [y_next]
        # self.s['u'] = [u_next]
        # self.s['v'] = [v_next]

        # Select some obstacles so that the computations are easier
        point = np.array([self.s['x'], self.s['y']]).reshape((-1, 2))
        if self.obstacles:
            self.culled = fast_distance(point, self.obstacle_centers) < 3 * PERCEPTION_DISTANCE

        # Get Observations
        observation = self.get_observation()

        # compute the reward
        reward = self._reward_function()

        # Result console logging
        if self.episode % self.logging_episodes == 0 and self.total_timestep == 1:
            print('Episode ', self.episode)
            print('good = ', self.n_done, ', timeouts = ', self.oo_time,
                  ', OoBounds = ', self.oob, ', Crashes = ', self.crashes)

            self.n_done = 0
            self.oo_time = 0
            self.oob = 0
            self.crashes = 0


        # has the game finished
        done = self.done

        # Render functions
        self.trajectory['x'].append(self.s['x'])
        self.trajectory['y'].append(self.s['y'])

        info = {}

        return observation, reward, np.array(done), info

    #TODO test
    # @profile
    def get_airsim_observation(self):
        as_pos = self.client.getPosition()
        as_vel = self.client.getVelocity()

        self.s['x'] = [as_pos.x_val]
        self.s['y'] = [as_pos.y_val]
        self.s['u'] = [as_vel.x_val]
        self.s['v'] = [as_vel.y_val]

        x, y, u, v, t_x, t_y = self.get_state()

        dist = np.linalg.norm((t_x - x, t_y - y))
        clipped_dist = min(MAX_DIST, dist) / MAX_DIST
        vel = np.linalg.norm(np.array([u, v]))
        clipped_vel = min(MAX_VEL, vel) / MAX_VEL

        dir_x = np.nan_to_num(((t_x - x) / (dist + 1e-6)) * clipped_dist)
        dir_y = np.nan_to_num(((t_y - y) / (dist + 1e-6)) * clipped_dist)

        vel_x = np.nan_to_num((u / (vel + 1e-6)) * clipped_vel)
        vel_y = np.nan_to_num((v / (vel + 1e-6)) * clipped_vel)

        # p_vector = self.grid_collisions_perception(x, y)
        p_vector = self.ray_collision_perception((x, y))
        # print('rays: {}'.format(p_vector))

        # if self.total_timestep % 20 == 0:
        #     print('state =        {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.2f}'.format(x, y, u, v, t_x, t_y))
        #     print('observations = {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}'.format(dir_x, dir_y, vel_x, vel_y))

        return np.concatenate(([dir_x, dir_y, vel_x, vel_y], p_vector))

    # TODO test
    def _airsim_action(self, action, x, y, u, v):
        """ discrete cartesian dynamics"""
        offset = self._interpret_action(action)

        vel = np.linalg.norm(np.array([u, v]))
        lim_u = np.abs(u / vel) * MAX_VEL
        lim_v = np.abs(v / vel) * MAX_VEL

        # x_next = x + u * self.time_step
        # y_next = y + v * self.time_step
        # u_next = (self.F * offset[0] - self.k * u) * self.time_step + u
        # v_next = (self.F * offset[1] - self.k * v) * self.time_step + v

        x_next = x
        y_next = y
        u_next = offset[0] + u
        v_next = offset[1] + v

        u_next = np.clip(u_next, -lim_u, lim_u)  # Limit the max speed
        v_next = np.clip(v_next, -lim_v, lim_v)

        self.client.moveByVelocity(u_next, v_next, 0, 10)

        # print(quad_vel.x_val + quad_offset[0], quad_vel.y_val + quad_offset[1],0, 20)
        time.sleep(1 / FRAMES_PER_SECOND)


        return x_next, y_next, u_next, v_next

    # TODO check collisions
    def _no_speed_reward(self):

        x, y, u, v, t_x, t_y = self.get_state()

        dist = np.float64(10000000)
        dist = min(dist, np.linalg.norm([x - t_x, y - t_y]))
        vel = np.linalg.norm(np.array([u, v]))

        # Reward in base of the distance
        # if self.total_timestep % 20 == 0:
        #     print('dist: ', np.array([dist]), ' vel: ', np.array([vel]))
        reward = 0
        if dist < self.threshold_dist:
            reward = + POSITIVE_REWARD
            self.done = np.array([True])
            # print('Target Found')
            self.n_done += 1
            self.episode_success = True
        else:
            d2 = np.square(dist)
            reward = (- 0.01 * (d2 + 1) - 80) * self.dense_reward
        # Max Timestep
        if self.total_timestep >= self.max_timestep:
            reward -= TIME_NEGATIVE_REWARD
            self.oo_time += 1
            self.done = np.array([True])
            # print('Timeout')

        # Out of bounds check
        if x < self.map_min_x or y < self.map_min_y or x > self.map_max_x or y > self.map_max_y:
            reward += -OUT_OF_BOUNDS_REWARD
            self.oob += 1
            self.done = np.array([True])
            # print('Out of Bounds')

        # COLLISIONS check
        for obstacle in IT.compress(self.prep_obstacles, self.culled):
            if sv.contains(obstacle, x=x, y=y):
                reward += -OUT_OF_BOUNDS_REWARD
                # print('Crash')
                self.crashes += 1
                self.done = np.array([True])
                break

        reward = reward / REWARD_NORMALIZATION_FACTOR
        # print(reward)

        # Printing functions

        self.trajectory['vel'].append(vel)
        self.trajectory['dist'].append(dist)
        self.trajectory['reward'].append(reward)

        return reward

    def move_to(self, x, y):
        self.client.moveToZ(z=-15, velocity=7)
        time.sleep(0.1)
        self.client.moveToPosition(x=x, y=y, z=-15, velocity=30,
                                   max_wait_seconds=60,
                                   drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=YawMode(), lookahead=-1,
                                   adaptive_lookahead=1)
        time.sleep(0.5)
        self.client.moveToPosition(x=x, y=y, z=-15, velocity=15,
                                   max_wait_seconds=60,
                                   drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=YawMode(), lookahead=-1,
                                   adaptive_lookahead=1)
        time.sleep(0.5)
        self.client.moveToPosition(x=x, y=y, z=7, velocity=2,
                                   max_wait_seconds=60,
                                   drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=YawMode(), lookahead=-1,
                                   adaptive_lookahead=1)
        time.sleep(0.2)
        self.client.moveToZ(z=7, velocity=2)
        time.sleep(0.1)
        return