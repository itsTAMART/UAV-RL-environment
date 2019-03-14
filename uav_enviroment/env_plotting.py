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
# import Box2D
# from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

MAX_DIST = 100
MAX_VEL = 50


# TODO make it faster
def render_trajectory(env):
    try:
        GAMMA = 0.99
        # [env.plot_obstacle(obstacle) for obstacle in env.obstacles]
        t = env.trajectory

        # If it's not empty
        if t['x']:
            x1 = np.arange(len(t['x']))

            # safe_size = len(env.latest_results) - (len(env.latest_results) % 5)
            # result_grid = np.array(env.latest_results)[0:safe_size].reshape((-1, 5)).T

            dr_sum = np.zeros(len(t['reward']))
            dr_sum[0] = t['reward'][0]
            for i in range(len(t['reward']) - 1):
                dr_sum[i + 1] = t['reward'][i + 1] + GAMMA * dr_sum[i]

            plt.clf()
            plt.figure(1)
            gs = gridspec.GridSpec(8, 12)

            # Trajectory
            ax1 = plt.subplot(gs[:, 0:8])
            # ax1 = plt.subplot()
            ax1.set(title='Trajectory of the UAV')
            [plot_obstacle(obstacle) for obstacle in env.obstacles]
            plt.scatter(env.s['origin_x'], env.s['origin_y'], color='g', label='origin')
            plt.scatter(env.s['target_x'], env.s['target_y'], color='r', label='target')
            plt.scatter(t['x'], t['y'], s=1, marker='x', color='black')
            plt.xlim(env.map_min_x, env.map_max_x)
            plt.ylim(env.map_min_y, env.map_max_y)
            # plt.title('Episode {}'.format(env.episode))
            plt.legend()
            plt.gca().set_aspect('equal', adjustable='box')

            # Speed and reward
            ax2 = plt.subplot(gs[:3, 8:])

            ax2.plot(x1, t['vel'], color='blue', label='speed')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.set(xlabel='Speed and reward.')
            ax2.yaxis.set_label_position("right")
            plt.ylim(-1, MAX_VEL + 1)
            ax2.yaxis.tick_right()
            # ax2.legend()

            ax21 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
            ax21.plot(x1, t['reward'], color='orange', label='reward')
            ax21.tick_params(axis='y', labelcolor='orange')
            ax21.set(ylabel='Reward')
            plt.ylim(-2, 1)
            # ax21.legend()

            # Cumulative Reward
            ax3 = plt.subplot(gs[3:6, 8:])
            ax3.plot(x1, dr_sum, 'k-', color='black', label='cumulative reward')
            ax3.set(xlabel='Cumulative Reward')
            ax3.yaxis.set_label_position("right")
            ax3.yaxis.tick_right()

            # # Last Results
            # ax4 = plt.subplot(gs[6:, 8:])
            # ax4.pcolor(result_grid, cmap='RdYlGn', vmin=-1.6, vmax=1.3)
            # ax4.set(xlabel='Last Results.')
            # ax4.set_xticks([])
            # ax4.set_yticks([])

            plt.tight_layout(pad=.5)
            plt.draw()
            plt.pause(0.0001)
    except (RuntimeError, TypeError, NameError):

        pass


def plot_obstacle(obstacle):
    x = []
    y = []
    for point in obstacle.exterior.coords:
        x.append(point[0])
        y.append(point[1])

    plt.fill(x, y, color='black', alpha=0.3)

#@profile
def obstacle_in_image(img, obstacle, pos, window):
    a1, _, a2 = obstacle.exterior.coords[0:3]
    #a2 = obstacle.exterior.coords[2]
    mid1 = a1 - pos + window
    mid2 = a2 - pos + window
    pt1 = (mid1).astype(np.int)
    pt2 = (mid2).astype(np.int)
    cv2.rectangle(img, tuple(pt1), tuple(pt2), (255, 0, 0), -1)
