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

PERCEPTION_DISTANCE = 40

def create_perception_matrix(dist=PERCEPTION_DISTANCE, n_radius: int = 8, pts_p_radius: int = 3):
    """Creates a matrix of offsets to add to the point and check if they intersect with the obstacles"""
    Z = np.zeros((2, n_radius, pts_p_radius))
    for i, angle in np.ndenumerate(np.arange(start=0, step=(2 / n_radius), stop=2)):
        Z[0, i, :] = np.cos(np.pi * angle)
        Z[1, i, :] = np.sin(np.pi * angle)
    long = np.arange(start=1, step=(dist / pts_p_radius), stop=dist).reshape(1, -1, pts_p_radius)
    Z = Z * long
    return Z.reshape((2, -1)).T


def create_perception_distances(max_dist=PERCEPTION_DISTANCE, n_radius: int = 16):
    """Creates a vector of lines and check if they intersect with the obstacles"""
    # TODO implement it
    return None


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
             angle_between((1, 0, 0), (0, 1, 0))
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_difference(p, v, t):  # posicion, velocidad y target calcula la diferencia de angulos
    a = np.arctan2(v[1], v[0]) - np.arctan2(t[1] - p[1], t[0] - p[0])  # Difference between 2 angles
    a = np.remainder(a + np.pi, 2 * np.pi) - np.pi  # we center the angle so that is between -pi and pi
    return a


def fast_distance(a, b):
    """
    Fast computation of distance using sqrt_einsum(self, x,y)
    :param a: points a as rows
    :param b: points b as rows
    :return: distance between a and b points
    """
    a_min_b = a - b
    return np.sqrt(np.einsum('ij,ij->i', a_min_b, a_min_b))

