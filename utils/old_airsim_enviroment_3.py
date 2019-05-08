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

path_to_airsimclient = '/home/daniel/Desktop/AIRSIM/AirSim-master/PythonClient/'

os.chdir(path_to_airsimclient)
from AirSimClient import *

np.set_printoptions(precision=2, suppress=True)
# client = None

"""
HARDCODINGS in HERE, change it to define other problem.

TODO not hardcode them

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
        [pos_x,pos_y,0,vel_x,vel_y,0,target_x,target_y,0,0,0]
        ACT:
        [accel_x,accel_y]

REWARD_NORMALIZATION_FACTOR = 1/this is the reward
FRAME_RATE = the fps at which the game will run
SCALING_FACTOR = the speed increase for the drone for each action
"""
# target_point = np.array([0, 0, 0])
thresh_dist = 20
REWARD_NORMALIZATION_FACTOR = 100
MAX_TIMESTEP = 500
TIME_NEGATIVE_REWARD = 0
OUT_OF_BOUNDS_REWARD = 1.2 * REWARD_NORMALIZATION_FACTOR * (6 * MAX_TIMESTEP + 60)
POSITIVE_REWARD = 1.2 * REWARD_NORMALIZATION_FACTOR * (2 * MAX_TIMESTEP)

# OBS:
# [pos_x,pos_y,0,vel_x,vel_y,0,target_x,target_y,0,0,0]
# ACT:
# [accel_x,accel_y]

ENV_OBS_LOWS = np.array([-55, -55, 0, -70, -70, 0, 0, 0, 0, 0, 0])
ENV_OBS_HIGH = np.array([250, 350, 0, 70, 70, 0, 140, 230, 0, 0, 0])
ENV_ACT_LOWS = np.array([-1, -1])
ENV_ACT_HIGH = np.array([1, 1])

FRAMES_PER_SECOND = 5
SCALING_FACTOR = 3


class Enviroment():
    """Class Enviroment for the AIRsim simulator
    This tries to recreate the OpenAI gym interface for the AIRsim simulator.

    """
    client = None
    total_timestep = 0
    done = False

    reward_range = (-float('inf'), float('inf'))
    action_space = None
    observation_space = None
    num_envs = 1

    def __init__(self):

        """"

        """

        self.done = False
        self.total_timestep = 0
        self.action_space = spaces.Box(low=ENV_ACT_LOWS, high=ENV_ACT_HIGH)
        self.observation_space = spaces.Box(low=ENV_OBS_LOWS, high=ENV_OBS_HIGH)
        self.num_envs = 1
        self.initial_pos = [0, 0, 0]
        self.target_point = self._generate_target_point()

    def setup(self):
        self.episode = 0
        self.client = MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPosition(x=0, y=0, z=0, velocity=30,
                                   max_wait_seconds=60,
                                   drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=YawMode(), lookahead=-1,
                                   adaptive_lookahead=1)
        self.client.hover()

    def reset(self):

        print('Reset, episode', self.episode, ' next target:', self.target_point)
        self.episode += 1
        self.total_timestep = 0
        self.client.moveToPosition(x=self.initial_pos[0], y=self.initial_pos[1], z=self.initial_pos[2], velocity=30,
                                   max_wait_seconds=60,
                                   drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=YawMode(), lookahead=-1,
                                   adaptive_lookahead=1)

        time.sleep(0.5)
        self.client.moveToPosition(x=self.initial_pos[0], y=self.initial_pos[1], z=self.initial_pos[2], velocity=15,
                                   max_wait_seconds=60,
                                   drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=YawMode(), lookahead=-1,
                                   adaptive_lookahead=1)
        time.sleep(0.5)
        self.client.moveToPosition(x=self.initial_pos[0], y=self.initial_pos[1], z=0, velocity=2,
                                   max_wait_seconds=60,
                                   drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=YawMode(), lookahead=-1,
                                   adaptive_lookahead=1)
        time.sleep(0.2)
        self.client.hover()
        self.done = False
        observation = self._get_observation()
        return observation

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

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
        if self.total_timestep % 20 == 0:
            print('TTL = ', self.total_timestep)
        # print( 'TTL = ', self.total_timestep)
        # print('action = ',action)
        # Take the action
        self._take_action(action)
        # Get Observations
        observation = self._get_observation()
        # compute the reward
        reward = self._compute_reward(observation)
        # has the game finished
        done = self.done
        if self.done: self.reset()
        # info = { 'episode': self.episode }
        info = {}
        return observation, reward, done, info

    def _get_observation(self):
        pos = self.client.getPosition()
        vel = self.client.getVelocity()
        # if self.total_timestep % 20 == 0:
        #     print('observations = ',
        #           np.array([pos.x_val, pos.y_val, vel.x_val, vel.y_val, self.target_point[0], self.target_point[1]]))
        return [pos.x_val, pos.y_val, 0, vel.x_val, vel.y_val, 0, self.target_point[0], self.target_point[1],
                self.target_point[2], 0, 0]

    # Compute the rewards and check the termination conditions
    def _compute_reward(self, observations):

        k1 = 0.01
        k2 = 1
        k3 = 0.01
        k4 = 1
        k5 = -80
        k6 = - k5
        k7 = 3
        k8 = 0.03

        dist = np.float64(10000000)

        # DISTANCE
        my_position = np.array([observations[0], observations[1], 0])
        dist = min(dist, np.linalg.norm(my_position - self.target_point))
        vel = np.linalg.norm(np.array([observations[2], observations[3], 0]))

        # Reward in base of distance
        # if self.total_timestep % 20 == 0:
        #     print('dist: ', np.array([dist]), ' vel: ', np.array([vel]))
        if dist < thresh_dist and vel < 1:
            reward = + POSITIVE_REWARD
            self.done = True
            self.initial_pos = self.target_point
            self.target_point = self._generate_target_point()
            print('Target Found, next target:', self.target_point)


        else:
            d2 = np.square(dist)
            v2 = np.square(vel)
            reward = -k1 * ((d2 + k4) * (1 + k2 * (np.min(k7 * v2 + k5, 0) / (k8 * d2 + k3)))) - k6
        # Max Timestep
        if self.total_timestep >= MAX_TIMESTEP:
            reward -= TIME_NEGATIVE_REWARD
            self.done = True
            print('MAX TIMESTEP Reached')

        # Out of bounds check
        if observations[0] < -50 or observations[1] < -50:
            reward += -OUT_OF_BOUNDS_REWARD
            print('Out of Bounds')
            self.done = True
        elif observations[0] > 250 or observations[1] > 350:
            reward += -OUT_OF_BOUNDS_REWARD
            print('Out of Bounds')
            self.done = True

        reward = reward / REWARD_NORMALIZATION_FACTOR

        # if self.total_timestep % 20 == 0:
        #     print('reward = ', np.array([reward]))
        return reward

    # #  termination conditions included in the rewad
    # def isDone(self, reward):
    #     self.done = False
    #     if reward == 0:
    #         self.done = True
    #     elif self.total_timestep >= MAX_TIMESTEP:
    #         self.done = True
    #     return self.done

    def _take_action(self, action):
        quad_offset = self._interpret_action(action)
        quad_vel = self.client.getVelocity()
        self.client.moveByVelocity(quad_vel.x_val + quad_offset[0],
                                   quad_vel.y_val + quad_offset[1],
                                   0, 20)

        # print(quad_vel.x_val + quad_offset[0], quad_vel.y_val + quad_offset[1],0, 20)
        time.sleep(1 / FRAMES_PER_SECOND)

    def _interpret_action(self, action):
        # if self.total_timestep % 20 == 0:
        #     print('action = ', action)
        scaling_factor = SCALING_FACTOR  # 3 originally
        quad_offset = (action[0][0] * scaling_factor, action[0][1] * scaling_factor, 0)
        return quad_offset

    def _generate_target_point(self):
        return np.multiply(np.random.uniform(size=3), np.array([140, 230, 0]))
