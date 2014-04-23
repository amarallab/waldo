#!/usr/bin/env python
'''
Filename: equally_space
Description: 
'''

__authors__ = 'Peter B. Winter and Andrea Lancanetti'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import math

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
shared_directory = project_directory + 'code/shared/'
assert os.path.exists(shared_directory)
sys.path.append(shared_directory)

# nonstandard imports
from thinning.distance import euclidean


def sign(x):
    x = float(x)
    if x == 0.:
        return 1.
    else:
        return math.fabs(x) / x

def correct_point(point_so_far, next_point, distance_from_next_point):
    """
        we correct point_so_far placing it along the line between point_so_far and next_point
        so that the distance_from_next_point is matched 
    """
    # if points have same x
    if point_so_far[0] == next_point[0]:
        return (point_so_far[0],
                next_point[1] - distance_from_next_point * sign(next_point[1] - point_so_far[1]))
    else:
        m = float(next_point[1] - point_so_far[1]) / (next_point[0] - point_so_far[0])
        x = next_point[0] - distance_from_next_point / math.sqrt(1 + m ** 2) * sign(next_point[0] - point_so_far[0])
        y = next_point[1] + m * (x - next_point[0])
        return (x, y)


def find_next_index(xy, next_index_along_original_line, point_so_far, step):
    """
        xy is a list of tuples (x,y)
        next_index_along_original_line is an integer, pointing to a point ahead
        point_so_far which is the last (x,y) in the equally space spine
    """
    dist = 0.
    while True:
        assert next_index_along_original_line < len(xy), 'find_next_index: BUG!'
        # next point along original line
        next_point = xy[next_index_along_original_line]
        # compute the distance we have walked so far plus the next point 
        dist += euclidean(point_so_far, next_point)

        # if this is further than what we want, stop
        if dist >= step:
            break
        # if this is not, point_so_far becomes next point and we move the next point further
        else:
        # if there are no more points to go further we stay where we are
            if next_index_along_original_line + 1 == len(xy):
                return dist, next_index_along_original_line, point_so_far
            next_index_along_original_line += 1
            point_so_far = next_point

    return dist, next_index_along_original_line, point_so_far



def equally_space_times(num_ids, step=0):
    """
        takes a list of times and returns an equally spaced ones
    """

    assert len(num_ids) > 1, 'not enough snapshots to equally space them in time'

    if not step:
        step = (num_ids[-1] - num_ids[0]) / float(len(num_ids) - 1)

    ids_eq = []
    t = num_ids[0]

    #print t, num_ids[0], num_ids[-1], step, '<<<<<<<<<<<<<'
    while t < num_ids[-1]:
        #print t
        ids_eq.append(t)
        t += step

    if len(ids_eq) < len(num_ids):
        ids_eq.append(num_ids[-1])

    ids_eq[-1] = num_ids[-1]
    #print len(num_ids), 'num_ids'
    #print len(ids_eq), 'ids_eq'
    #print ids_eq[-2], num_ids[-2]
    assert len(num_ids) == len(ids_eq), 'BUG! equally spaced times do not have the same length'
    return ids_eq



def linear_interpolation(x, point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        return y1
    else:
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1

def fixed_step_equally_space_times(times, step=0.1, ndigits=1):
    assert len(times) > 1, 'not enough snapshots to equally space them in time'
    start_time = round(times[0], ndigits=ndigits)
    end_time = round(times[-1], ndigits=ndigits)
    N = int((end_time - start_time) / float(step))
    #print start_time, end_time, N
    fixed_times = map(lambda x: round(step * x, n, range(N)))
    #print fixed_times
    return fixed_times

if __name__ == '__main__':
    t = range(0, 100)
    t = map(lambda x: 0.1 * x, t)
    import random
    def add_noise(x):
        m = 1
        noise = (random.uniform(0, m) - 0.5 * m) * 0.01
        #x * 0.1 , noise * 0.01
        return x + noise
    t = map(add_noise, t)
    #print equally_space_times(t, step=0.1)
    fixed_step_equally_space_times(t)
