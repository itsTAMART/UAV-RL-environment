import numpy as np
from shapely.geometry import Point, asPoint, asPolygon

offset = [500, -500, -500, 500]

obstacles = \
    [[11500, -11750, 10500, -10750],
     [-2500, -11750, -3500, -10750],
     [-1500, -5750, -1500, -5750],
     [8500, -6500, 7500, -5750],
     [3500, -3750, 2500, -2750],
     [11500, -1750, 8500, 1250],
     [3500, 2250, 2500, 3250],
     [-1500, 5250, -1500, 5250],
     [8500, 5250, 7500, 6250],
     [11500, 10250, 10500, 11250],
     [-2500, 10250, -3500, 11250]]

points_obs = \
    [[12000, -12250, 10000, -10250],
     [-2000, -12250, -4000, -10250],
     [-1000, -6250, -2000, -5250],
     [9000, -7000, 7000, -5250],
     [4000, -4250, 2000, -2250],
     [12000, -2250, 8000, 1750],
     [4000, 1750, 2000, 3750],
     [-1000, 4750, -2000, 5750],
     [9000, 4750, 7000, 6750],
     [12000, 9750, 10000, 11750],
     [-2000, 9750, -4000, 11750]]

offset = np.array(offset)
obstacles = np.array(obstacles)

sx = 160 / (12000 - (-4000))
sy = 240 / (11750 - (-12250))

ox = +4000
oy = +12250

point_offset = [ox, oy, ox, oy]
point_scale = [sx, sy, sx, sy]

points_obs = obstacles + offset

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

scaled_points = (points_obs + np.array(point_offset)) * np.array(point_scale)

orig_start = [0, 0]

start = (np.array(orig_start) + np.array(point_offset[0:2])) * np.array(point_scale[0:2])

# start = [ 35.        , 117.39583333] 


objectives = \
    [[260, 8280],
     [-2640, -150],
     [11310, 6400],
     [3060, -10970],
     [10720, -6310],
     [5050, 3570]]

scaled_objectives = (np.array(objectives) + np.array(point_offset[0:2])) * np.array(point_scale[0:2])

scaled_objectives = \
    [[42.6, 205.3],
     [13.6, 121.],
     [153.1, 186.5],
     [70.6, 12.8],
     [147.2, 59.4],
     [90.5, 158.2]]

obstacle_polygon = []

for obstacle in scaled_points:
    x0, y0 = obstacle[2], obstacle[1]
    w = obstacle[0] - obstacle[2]
    h = obstacle[3] - obstacle[1]

    p1, p2, p3, p4 = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
    # print([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)])
    obstacle_polygon.append(asPolygon([p1, p2, p3, p4]))

# x0, y0 = obstacle[2], obstacle[1]
# w = obstacle[0] - obstacle[2]
# h = obstacle[3] - obstacle[1]
