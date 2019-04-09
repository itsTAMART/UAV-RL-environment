import cv2
import time
import math
import sys
import os
import gym
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

import gym
from gym import spaces
from gym.utils import seeding, EzPickle


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

VIEWPORT_W = 600
VIEWPORT_H = 400

"""
HARDCODINGS in HERE, change it to define other problem.

Target Point: the target point you try to reach
threshold_dist = the distance at which we consider the game done
MAX_TIMESTEP = constant with maximum amount of steps to reach the target
TIME_NEGATIVE_REWARD = When you exceed the Max timestamp you get a negative reward 
(this number was 130(max_x)^2 + 230(max_y)^2)
OUT_OF_BOUNDS_REWARD = when you get out of bounds you get a small negative reward
POSITIVE_REWARD = reward when you reach the goal

inecuations for reward shaping:
    POSITIVE_REWARD > 2*MAX_TIMESTEP
    OUT_OF_BOUNDS_REWARD > 6 *MAX_TIMESTEP + 60(this numbers depend on the scaling factor)


ENV_LOWS and ENV_HIGHS are the constants to define the space, the first 2 components are the range in which we can move
the last 2 are the min and max speed.
    Args:
        OBS: 
        [pos_x,pos_y,vel_x,vel_y,target_x,target_y]
        ACT:
        [accel_x,accel_y]

REWARD_NORMALIZATION_FACTOR = 1/this is the reward
FRAME_RATE = the fps at which the game will run
SCALING_FACTOR = the speed increase for the drone for each action
"""

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

# # TODO go back to 64 it
OBS_IMAGE_WIDTH = 600
OBS_IMAGE_HEIGTH = 600

STEERING_VEL = 0.1

FRAMES_PER_SECOND = 5
SCALING_FACTOR = 4
GAMMA = 0.99
RENDER_EPISODE = 5


# Helper Classes if needed


class UAVEnv(gym.Env):
    """ Class UAVEnv that models the AIRsim simulator dynamics.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    continuous = True  # Denotes if the action space is continuous
    dense_reward = True  # True if the reard is dense, else it will be +1 if success -1 if not
    angular_movement = True  # When True dynamics are modeled with steer and thrust, else they are modeled with dx, dy
    observation_with_image = False  # When True return an image alongside the observation
    reset_always = True  # When True change the environment origin, target and obstacles every time it resets.
    controlled_speed = True  # When True the task is to reach the target with a controlled speed, else is just reach the target

    logging_episodes = 1000

    map_size_x = 200
    map_size_y = 200
    map_min_x = - map_size_x * 0.1
    map_min_y = - map_size_y * 0.1
    map_max_x = map_size_x * 1.1
    map_max_y = map_size_y * 1.1
    threshold_dist = 40
    threshold_vel = 4

    n_obstacles = 0

    max_timestep = 1000

    def __init__(self, continuous=True, angular_movement=True, observation_with_image=False, reset_always=True,
                 controlled_speed=True):
        self.seed()
        self.viewer = None

        self.continuous = continuous  # Denotes if the action space is continuous
        # dense_reward = True  # True if the reard is dense, else it will be +1 if success -1 if not
        self.angular_movement = angular_movement  # When True dynamics are modeled with steer and thrust, else they are modeled with dx, dy
        self.observation_with_image = observation_with_image  # When True return an image alongside the observation
        self.reset_always = reset_always  # When True change the environment origin, target and obstacles every time it resets.
        self.controlled_speed = controlled_speed  # When True the task is to reach the target with a controlled speed, else is just reach the target
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

        # Model the angular_perception according to the dynamics: either ANGULAR or CARTESIAN
        if self.angular_movement:
            n_discrete_actions = 4  # Thrust, Brake,  Steer Left, Steer Right
            env_act_lows = np.array([0, -1])  # Thrust and Steer
            env_act_high = np.array([1, 1])
        else:
            # Perception matrix initialization, only with cartesian(dx,dy) dynamics.
            self.perception_matrix = create_perception_matrix(dist=PERCEPTION_DISTANCE, n_radius=16, pts_p_radius=6)
            self.perception_ray_points = create_perception_distance_points(max_dist=PERCEPTION_DISTANCE, n_radius=16)
            n_discrete_actions = 4  # +dx, -dx, +dy, -dy
            env_act_lows = np.array([-1, -1])  # dx and dy
            env_act_high = np.array([1, 1])

        # Define the action space either CONTINUOUS or DISCRETE.
        if self.continuous:
            self.action_space = spaces.Box(low=env_act_lows, high=env_act_high)
        else:
            self.action_space = spaces.Discrete(n_discrete_actions)

        # Define the action space either CONTINUOUS or DISCRETE.
        if self.angular_movement:
            ENV_OBS_LOWS = np.zeros(shape=self.get_angular_observation().shape)
            ENV_OBS_HIGH = np.ones(shape=self.get_angular_observation().shape)
        else:
            ENV_OBS_LOWS = np.zeros(shape=self.get_cartesian_observation().shape)
            ENV_OBS_HIGH = np.ones(shape=self.get_cartesian_observation().shape)
            # ENV_OBS_HIGH[0:6] = [self.map_max_x, self.map_max_y, MAX_VEL, MAX_VEL, self.map_max_x, self.map_max_y]

        # self.observation_space = spaces.Box(low=ENV_OBS_LOWS, high=ENV_OBS_HIGH, dtype=np.float32)

        # Define the observation space with or without an image
        if self.observation_with_image:
            self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=observation_shape)
        else:
            self.observation_space = spaces.Box(low=ENV_OBS_LOWS, high=ENV_OBS_HIGH, dtype=np.float32)

        if self.controlled_speed:
            self.reward_function = self._controlled_speed_reward()  # controlled speed reward
        else:
            self.reward_function = self._no_speed_reward()  # no speed reward

    def setup(self, *, map_size_x=map_size_x, map_size_y=map_size_y, n_obstacles=0, reset_always=True,
              max_timestep=1000, threshold_dist=40, threshold_vel=4, reward_sparsity=False):

        self.map_size_x = map_size_x
        self.map_size_y = map_size_y
        self.map_min_x = - map_size_x * 0.1
        self.map_min_y = - map_size_y * 0.1
        self.map_max_x = map_size_x * 1.1
        self.map_max_y = map_size_y * 1.1
        self.n_obstacles = n_obstacles
        self.reset_always = reset_always
        self.max_timestep = max_timestep
        self.threshold_dist = threshold_dist
        self.threshold_vel = threshold_vel
        # True is sparse reward, False is a dense reward
        self.reward_sparsity = reward_sparsity
        self.dense_reward = not reward_sparsity

        if self.angular_movement:  # set the appropriate methods for the observation
            self.get_observation = self.get_angular_observation
        else:
            self.get_observation = self.get_cartesian_observation

        if self.observation_with_image:
            self.get_observation = self.get_image_observation

        if self.continuous:  # For continuous actions
            if self.angular_movement:
                self._take_action = self._cont_angular_action  # Continuous angular
            else:
                self._take_action = self._cont_cartesian_action  # Continuous cartesian
        else:  # For discrete actions
            if self.angular_movement:
                self._take_action = self._disc_angular_action  # Discrete angular
            else:
                self._take_action = self._disc_cartesian_action  # Discrete cartesian

        self.generate_map()

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        # print('Reset, episode', self.episode, ' next target:', self.target)
        self.episode += 1
        self.total_timestep = 0
        self.trajectory = copy.deepcopy(EMPTY_TRAJECTORY)

        # # TODO remove after
        # self.output_video = None
        # if self.RENDER_FLAG:
        #     if self.output_video is not None:
        #         self.output_video.release()
        #     # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        #     self.output_video = cv2.VideoWriter('/home/daniel/Videos/DQN_training/DQN_ep{}.avi'.format(self.episode),
        #                                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60,
        #                                         (IMAGE_WIDTH, IMAGE_HEIGTH))
        #

        # NEW ENVIRONMENT
        if self.reset_always:
            # Reset the initial position and the target ALWAYS
            self.generate_map()
            self.episode_success = False
        else:
            # Reset the initial position and the target WHEN SUCCESSFUL
            if self.episode_success:
                self.generate_map()
                self.episode_success = False

        # Origin and target are modified in the generate map function
        self.s['x'] = self.s['origin_x']
        self.s['y'] = self.s['origin_y']
        self.s['u'] = [np.multiply(np.random.uniform(size=1), 0.1)[0] - 0.05]
        self.s['v'] = [np.multiply(np.random.uniform(size=1), 0.1)[0] - 0.05]

        observation = self.get_observation()
        self.done = np.array([False])

        return observation

    # @profile
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
        # Take the action
        x, y, u, v, t_x, t_y = self.get_state()

        x_next, y_next, u_next, v_next = self._take_action(action, x, y, u, v)

        self.s['x'] = [x_next]
        self.s['y'] = [y_next]
        self.s['u'] = [u_next]
        self.s['v'] = [v_next]

        # Select some obstacles so that the computations are easier
        point = np.array([self.s['x'], self.s['y']]).reshape((-1, 2))
        if self.obstacles:
            self.culled = fast_distance(point, self.obstacle_centers) < 3 * PERCEPTION_DISTANCE

        # Get Observations
        observation = self.get_observation()

        # # TODO remove after
        # if self.RENDER_FLAG:
        #     img = self.get_image_observation()
        #     cv2.imshow('observation', img)
        #     cv2.waitKey(1)
        #     if self.output_video is not None:
        #         self.output_video.write(img)

        # compute the reward
        reward = self._reward_function()
        # has the game finished
        done = self.done

        # Render functions
        self.trajectory['x'].append(self.s['x'])
        self.trajectory['y'].append(self.s['y'])

        info = {}

        return observation, reward, np.array(done), info

    def get_observation(self):
        """ method called to get the observation, override with the appropriate one """
        return self.get_angular_observation()

    def get_state(self):
        x = self.s['x'][0]
        y = self.s['y'][0]
        u = self.s['u'][0]
        v = self.s['v'][0]
        t_x = self.s['target_x'][0]
        t_y = self.s['target_y'][0]

        return x, y, u, v, t_x, t_y

    # @profile
    def get_cartesian_observation(self):
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

    def grid_collisions_perception(self, x, y):
        point = np.array([x, y]).reshape((-1, 2))
        PER = self.perception_matrix + point
        p_vector = np.zeros(PER.size // 2)

        # get only close objects to check the perception
        # culled = [as_point.distance(obstacle) < 1.5*PERCEPTION_DISTANCE for obstacle in self.obstacles]
        if self.obstacles:
            self.culled = fast_distance(point, self.obstacle_centers) < 3 * PERCEPTION_DISTANCE
            for obstacle in IT.compress(self.prep_obstacles, self.culled):
                # Vectorized
                p_i = sv.contains(obstacle, x=PER[:, 0], y=PER[:, 1])
                p_vector = np.logical_or(p_vector, p_i)

        return p_vector

    def ray_collision_perception(self, position):
        """

        :param position: numpy array with the position of the agent
        :return: a list with distances to the collision of each one of the rays
        """
        rays = create_perception_rays(position=position, points=self.perception_ray_points)
        position_point = np.array(position).reshape((-1, 2))
        # get only close objects to check the perception
        if self.obstacles:
            self.culled = fast_distance(position_point, self.obstacle_centers) < 3 * PERCEPTION_DISTANCE
            for obstacle in IT.compress(self.prep_obstacles, self.culled):
                # Vectorized
                rays = [ray.difference(obstacle) for ray in rays]
        # TODO test

        # print('rays: {}'.format(rays))
        return [(ray.length / PERCEPTION_DISTANCE) for ray in rays]


    def get_angular_observation(self):
        x, y, u, v, t_x, t_y = self.get_state()
        # This gets the angle between the direction of your speed and the direction of the target
        dist = min(MAX_DIST, np.linalg.norm((t_x - x, t_y - y))) / MAX_DIST
        theta = np.nan_to_num(angle_difference([x, y], [u, v], [t_x, t_y]))
        vel = min(MAX_VEL, np.sqrt(u ** 2 + v ** 2)) / MAX_VEL
        p_choque, theta_crash = self.angular_perception()

        return np.array([dist, theta, vel, p_choque, theta_crash])

    def angular_perception(self):
        point = np.array([self.s['x'], self.s['y']])
        vel = np.array([self.s['u'], self.s['v']])

        # get only close objects to check the angular_perception
        if len(self.obstacles) > 0:
            self.culled = fast_distance(point, self.obstacle_centers) < 3 * PERCEPTION_DISTANCE

        crash_angle = 0
        p_crash = 0
        left_angle = 0
        right_angle = 0
        for obstacle in IT.compress(self.prep_obstacles, self.culled):
            exteriors = np.array(obstacle.exterior.coords.xy)
            # Angles between our speed and the obstacles borders
            obstacle_angles = angle_difference(point, vel, exteriors)
            if not (all(a < 0 for a in obstacle_angles[0:4]) or
                    all(a > 0 for a in obstacle_angles[0:4]) or
                    all(a > np.pi / 2 for a in np.abs(obstacle_angles[0:4]))):
                # print('crush')
                # We care for the maximum angle of each side
                # Left angles are the ones with a positive difference
                candidate_left_angle = max([x for x in obstacle_angles if x > 0])
                # Right angles are the ones with a negative difference
                candidate_right_angle = min([x for x in obstacle_angles if x <= 0])

                if candidate_left_angle > left_angle: left_angle = candidate_left_angle
                if candidate_right_angle < right_angle: right_angle = candidate_right_angle
                p_crash = max(p_crash,
                              1 - (min(fast_distance(point.reshape(2, 1), exteriors[0:4])) / (2 * PERCEPTION_DISTANCE)))

            candidate_angles = np.array([left_angle, right_angle])
            crash_angle = candidate_angles[np.argmin(np.abs(candidate_angles))]
            # sigue siendo un poco naive la implementacion pero coges el angulo mas grande para desviarte, dando igual izquierda o derecha
            # dará problemas pero hay que encontrar una forma de manejar los angulos positivos de una forma y los negativos de otra.
        # print('crash angle: ', crash_angle, ' p_crash: ', p_crash)

        return p_crash, crash_angle

    # @profile
    def get_image_observation(self):
        x, y, u, v, t_x, t_y = self.get_state()

        img = self.field_of_view.copy()
        pos = np.array((x, y))
        half_window = np.array([IMAGE_WIDTH // 2, IMAGE_HEIGTH // 2])

        # Plot target
        target = np.array((t_x, t_y)) - pos + half_window
        cv2.circle(img, tuple((target).astype(np.int)), self.threshold_dist, (0, 0, 255), thickness=-1)

        # Plot obstacles

        # for obstacle in self.obstacles:
        for obstacle in IT.compress(self.prep_obstacles, self.culled):
            obstacle_in_image(img, obstacle, pos, half_window)

        # Plot the ship and velocity
        u = u * 4
        v = v * 4
        cv2.circle(img, tuple(half_window), 4, (0, 255, 0), thickness=-1)
        cv2.line(img, tuple(half_window), (int(IMAGE_WIDTH // 2 + u), int(IMAGE_HEIGTH // 2 + v)), (0, 255, 0), 2)

        # # Print Equivalent
        # cv2.imshow('observation', img)
        # cv2.waitKey(1)
        #

        img = cv2.resize(img, (OBS_IMAGE_HEIGTH, OBS_IMAGE_WIDTH))
        # print('img')
        return img

    def _take_action(self, action):
        """ method called to modify the state with each action, override with the appropriate one"""
        return self._cont_angular_action(action)

    def _cont_angular_action(self, action, x, y, u, v):
        """ continuous angular dynamics"""

        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        r = np.sqrt(u ** 2 + v ** 2)
        theta = np.arctan2(v, u)
        r_next = min(MAX_VEL, max(0, r + (self.F * action[0] - self.k * r)))
        theta_next = theta + action[1] * np.pi * STEERING_VEL
        u_next = r_next * np.cos(theta_next)
        v_next = r_next * np.sin(theta_next)

        return x_next, y_next, u_next, v_next

    def _cont_cartesian_action(self, action, x, y, u, v):
        """ continuous cartesian dynamics"""
        vel = np.linalg.norm(np.array([u, v]))
        lim_u = np.abs(u / vel) * MAX_VEL
        lim_v = np.abs(v / vel) * MAX_VEL
        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        u_next = (self.F * action[0] - self.k * u) * self.time_step + u
        v_next = (self.F * action[1] - self.k * v) * self.time_step + v
        u_next = np.clip(u_next, -lim_u, lim_u)  # Limit the max speed
        v_next = np.clip(v_next, -lim_v, lim_v)

        return x_next, y_next, u_next, v_next

    def _disc_angular_action(self, action, x, y, u, v):
        """ discrete angular dynamics"""
        offset = self._interpret_action(action)

        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        r = np.sqrt(u ** 2 + v ** 2)
        theta = np.arctan2(v, u)
        r_next = min(MAX_VEL, max(0, r + (self.F * offset[0] - self.k * r)))
        theta_next = theta + offset[1] * np.pi * STEERING_VEL
        u_next = r_next * np.cos(theta_next)
        v_next = r_next * np.sin(theta_next)

        return x_next, y_next, u_next, v_next

    def _disc_cartesian_action(self, action, x, y, u, v):
        """ discrete cartesian dynamics"""
        offset = self._interpret_action(action)

        vel = np.linalg.norm(np.array([u, v]))
        lim_u = np.abs(u / vel) * MAX_VEL
        lim_v = np.abs(v / vel) * MAX_VEL
        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        u_next = (self.F * offset[0] - self.k * u) * self.time_step + u
        v_next = (self.F * offset[1] - self.k * v) * self.time_step + v
        u_next = np.clip(u_next, -lim_u, lim_u)  # Limit the max speed
        v_next = np.clip(v_next, -lim_v, lim_v)

        return x_next, y_next, u_next, v_next

    def _interpret_action(self, action):
        # print(action)
        scaling_factor = 1
        if action == 0:
            quad_offset = (scaling_factor, 0)
        elif action == 1:
            quad_offset = (-scaling_factor, 0)
        elif action == 2:
            quad_offset = (0, -scaling_factor)
        elif action == 3:
            quad_offset = (0, scaling_factor)
        else:
            quad_offset = (0, 0)
        return quad_offset

    def _reward_function(self):
        """ method called to compute the reward based on which type of task it is:
                eg: reach the target or reach it with controlled speed

        ALSO logging results

        """
        if self.episode % self.logging_episodes == 0 and self.total_timestep == 1:
            print('Episode ', self.episode)
            print('good = ', self.n_done, ', timeouts = ', self.oo_time, ', OoBounds = ', self.oob, ', Crashes = ',
                  self.crashes)

            self.n_done = 0
            self.oo_time = 0
            self.oob = 0
            self.crashes = 0

        return self._controlled_speed_reward()

    # @profile
    def _controlled_speed_reward(self):
        # k1 = 0.01
        # k2 = 1
        # k3 = 0.01
        # k4 = 1
        # k5 = -80
        # k6 = - k5
        # k7 = 3
        # k8 = 0.03
        # # reward = -k1 * ((d2 + k4) * (1 + k2 * (min(k7 * v2 + k5, 0) / (k8 * d2 + k3)))) - k6

        x, y, u, v, t_x, t_y = self.get_state()

        dist = np.float64(10000000)
        dist = min(dist, np.linalg.norm([x - t_x, y - t_y]))
        vel = np.linalg.norm(np.array([u, v]))

        # Reward in base of distance
        # if self.total_timestep % 20 == 0:
        #     print('dist: ', np.array([dist]), ' vel: ', np.array([vel]))
        reward = 0
        if dist < self.threshold_dist and vel < self.threshold_vel:
            reward = + POSITIVE_REWARD
            self.done = np.array([True])
            # print('Target Found')
            self.n_done += 1
            self.episode_success = True
        else:
            d2 = np.square(dist)
            v2 = np.square(vel)
            reward = (-0.01 * (
                        (d2 + 1) * (1 + 1 * (min(3 * v2 - 80, 0) / (0.03 * d2 + 0.01)))) - 80) * self.dense_reward
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

    def _no_speed_reward(self):

        x, y, u, v, t_x, t_y = self.get_state()

        dist = np.float64(10000000)
        dist = min(dist, np.linalg.norm([x - t_x, y - t_y]))
        vel = np.linalg.norm(np.array([u, v]))

        # Reward in base of distance
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

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if mode == 'human':
            render_trajectory(self)
        if mode == 'rgb_array':
            return self.get_image_observation()
        pass

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        super().close()

        cv2.destroyAllWindows()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate_map(self):
        self.obstacles = []
        for _ in range(self.n_obstacles):
            self.add_obstacle(self.generate_obstacle())
        # Added these 2 lines to reduce number of polys and increase performance
        # self.obstacles = cascaded_union(self.obstacles)
        # self.prep_obstacles = [prep(polygon) for polygon in self.obstacles]
        self.prep_obstacles = self.obstacles
        # self.obstacle_centers = np.array([obstacle.centroid.coords[0] for obstacle in self.obstacles])
        self.s['origin_x'], self.s['origin_y'] = self.generate_start_point()
        self.s['target_x'], self.s['target_y'] = self.generate_target_point()

    def generate_start_point(self):
        intersections = [True]  # We initialize it with something to run the loop
        origin = [0, 0]

        while any(intersections):
            origin = np.multiply(np.random.uniform(size=2), np.array([self.map_size_x, self.map_size_y]))
            point = asPoint(origin)
            intersections = []
            [intersections.append(obstacle.contains(point)) for obstacle in self.obstacles]

        return [origin[0]], [origin[1]]

    def generate_target_point(self):
        intersections = [True]  # We initialize it with something to run the loop
        target = [0, 0]
        while any(intersections):
            dist = 0
            while dist < 2 * self.threshold_dist:
                target = np.multiply(np.random.uniform(size=2), np.array([self.map_size_x, self.map_size_y]))
                dist = np.linalg.norm([self.s['origin_x'] - target[0], self.s['origin_y'] - target[1]])

            point = asPoint(target)
            intersections = []
            [intersections.append(obstacle.contains(point)) for obstacle in self.obstacles]

        return [target[0]], [target[1]]

    def generate_obstacle(self):
        """
        :return: a rectangle obstacle
        """
        x0 = np.random.rand() * self.map_size_x * 0.9
        y0 = np.random.rand() * self.map_size_y * 0.9
        w = (np.random.rand() + 0.2) * self.map_size_x * 0.2
        h = (np.random.rand() + 0.2) * self.map_size_y * 0.2
        p1, p2, p3, p4 = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
        # print([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)])
        return asPolygon([p1, p2, p3, p4])

    def add_obstacle(self, obstacle: Polygon):
        self.obstacles.append(obstacle)
        self.obstacle_centers = np.array([obstacle.centroid.coords[0] for obstacle in self.obstacles])
        self.prep_obstacles = self.obstacles
