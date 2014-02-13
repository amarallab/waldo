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
from GeometricCalculations.distance import euclidean

def compute_transpose(x):
    """ given a matrix, computes the transpose """
    xt=[([0]*len(x)) for k in x[0]]
    for i, x_row in enumerate(x):
        for j, b in enumerate(x_row):
            xt[j][i]=x[i][j]
    return xt

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


def equally_space(xy, points=-1, prefixed_step='no_prefixed_step', verbose=False, forgiving=True):
    """
        this functions takes a list xy
        in the format [(x1, y1), (x2, y2)...]
        points is the number of x, y pairs desired
    """
    if forgiving and len(xy) <= 2:
        return []

    assert len(xy) > 2, 'equally_space2 function found a very short spine. this is a BUG!'
    if points == -1:
        points = len(xy)
        #print 'len xy', len(xy)
    tot_distance = 0
    for k, a in enumerate(xy[:-1]):
        #tot_distance += euclidean_distance(xy[k][0], xy[k][1], xy[k + 1][0], xy[k + 1][1])
        tot_distance += euclidean(xy[k], xy[k + 1])
        #print tot_distance, "total distance"
    step = tot_distance / (points - 1)
    #print step, "step"

    if prefixed_step != 'no_prefixed_step':
        step = prefixed_step
        points = int(tot_distance / step)

    if verbose:
        print 'prefixed_step', prefixed_step
        print 'tot_distance', tot_distance
        print 'num points', points
        print 'step size', step

    equally_spaced = [(xy[0][0], xy[0][1])]
    next_index_along_original_line = 1

    for k in xrange(points - 1):
        point_so_far = equally_spaced[-1]
        # find the next point along xy which is further than step
        dist, \
        next_index_along_original_line, \
        point_so_far = find_next_index(xy, next_index_along_original_line,
                                       point_so_far, step)

        # correct the point decreasing its dictance from the next one
        correction = dist - step
        point_so_far = correct_point(point_so_far, xy[next_index_along_original_line], correction)
        equally_spaced.append(point_so_far)

    #print len(equally_spaced), 'equally spaced'

    return equally_spaced

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

def equally_space_snapshots_in_time(ids_strings, original_snapshots):
    """
        ids_strings is a list of strings with the times of the snapshots
        original_snapshots is a list of list, so that original_snapshots[i] looks like [x1, y1, x2, y2, ...]
        returns equally spaced times and snapshots
    """

    num_ids = [float(v) for v in ids_strings]
    ids_eq = equally_space_times(num_ids)

    tsnapshots = compute_transpose(original_snapshots)
    equally_spaced_snapshots = []

    for tsnapshot in tsnapshots:
        eq_values = equally_space_snapshots_in_time_1d(num_ids, ids_eq, tsnapshot)
        equally_spaced_snapshots.append(eq_values)

    ids_strings_eq = [str(v) for v in ids_eq]

    return ids_strings_eq, compute_transpose(equally_spaced_snapshots)


def equally_space_snapshots_in_time_1d(ids_not_eq, ids_eq, values, interpolation_kind='linear'):
    """
        ids_not_eq and ids_eq are lists of floats with times
        values is a list of floats you want to equally space in time
        interpolation_kind should be 'linear'. nothing more for now.
        returns equally spaced positions
    """

    assert ids_not_eq == sorted(ids_not_eq), 'DATA PROBLEM: time ids are not sorted!'
    assert ids_eq == sorted(ids_eq), 'BUG!!! ids_eq not sorted!!!'

    import bisect

    eq_values = []
    for k, dummy in enumerate(ids_eq):

        left_index = bisect.bisect_left(ids_not_eq, ids_eq[k])
        assert left_index < len(ids_not_eq), 'left index is out of range'

        if ids_not_eq[left_index] == ids_eq[k]:
            eq_values.append(values[left_index])

        else:
            left_index -= 1
            right_index = left_index + 1
            #print left_index, right_index, len(ids_eq), len(ids_not_eq), len(values)
            eq_values.append(linear_interpolation( \
                ids_eq[k],
                (ids_not_eq[left_index], values[left_index]), \
                (ids_not_eq[right_index], values[right_index])) \
                )

            #print ids_eq[k], ids_not_eq[left_index], ids_not_eq[right_index], 'III', left_index, right_index, k
            #print eq_values[-1], values[left_index], values[right_index]
            assert ids_eq[k] >= ids_not_eq[left_index] and ids_eq[k] < ids_not_eq[right_index], 'bisection is wrong'

    return eq_values


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
