#!/usr/bin/env python

'''
Filename: measurement_switchboard.py
Description: 
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import math
from itertools import izip

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(HERE + '/../../')
CODE_DIR = os.path.abspath(HERE + '/../')
SHARED_DIR = CODE_DIR + 'shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

from GeometricCalculations.distance import euclidean
from wio.file_manager import get_blob_ids, get_data

# TODO? move remove flag functionality further upstream?
def pull_basic_data_type(blob_id, data_type, remove_flags=True, **kwargs):    
    ''' default option to remove flagged timepoints!'''
    times, data = get_data(blob_id, data_type=data_type, **kwargs)
    if remove_flags:
        times_f, all_flags = get_data(blob_id, data_type='flags', **kwargs)
        flags = consolidate_flags(all_flags)
        unflagged_timeseries = [(t, s) for (t, s, f) in izip(times, data, flags) if f]
        times, data = zip(*unflagged_timeseries)
    return times, data

# todo
def space_basic_data_type_linearly(times, values):
    return times, values

def compute_size(blob_id, **kwargs):    
    # size is a body shape property. if we've flagged the shape, don't measure size.
    return pull_basic_data_type(blob_id, data_type='size_raw',
                                remove_flags=True, **kwargs)

def compute_width(blob_id, **kwargs):
    # size is a body shape property. if we've flagged the shape, don't measure size.
    return pull_basic_data_type(blob_id, data_type='width50', 
                                remove_flags=True, **kwargs)

def compute_centroid_speed(blob_id, **kwargs):
    # speed is not a body shape.
    # TODO get xy_raw processed ahead of time.
    times, xy, _ = get_data(blob_id, data_type='xy_raw', **kwargs)    
    stimes, speeds = [], []
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        xy_future = xy[i + 1]
        xy_now = xy[i]
        ds = math.sqrt((xy_future[0] - xy_now[0]) ** 2 + (xy_future[1] - xy_now[1]) ** 2)
        # in case of divide by zero error
        if dt > 0.0000001:
            stimes.append(t[i])
            speeds.append(ds/dt)
    return times, speeds

def length_of_spine(spine):
        dx, dy = map(np.diff, map(np.array, zip(*spine)))
        return np.sqrt(dx**2 + dy**2).sum()

def speed_along_spine(spine1, spine2, t_plus_dt, t, perpendicular, points='all'):
    '''
        this function takes a spine
        list of [[x1,y1], [x2,y2], ...]
        and returns the average speed along (or perpendicular to) the spine
    '''
    dt = t_plus_dt - t
    assert dt > 0, 'times are not sorted!'
    assert len(spine1) > 0, 'spine1 is empty'
    assert len(spine1) == len(spine2), 'spines of different lengths'
    if points == 'all': points = range(len(spine1))
    displacement = displacement_along_curve(spine1, spine2, perpendicular=perpendicular, 
                                                    points=points)
    speed = displacement / len(points) / dt
    return speed

# TODO: rewrite using numpy
def displacement_along_curve(curve1, curve2, perpendicular=False, points='all'):
    """
        curve1 and curve2 are lists of tuples (x,y)
        the function return the sum of the 
        projections of (curve2-curve1) along curve1
        if perpendicular, the displacement is computed
        perpendicular to the spine
    """
    if points == 'all': points = range(len(curve1))
    total_displacement = 0.
    for i in points:
        if i + 1 < len(curve1):
            p2 = curve1[i + 1]
            p1 = curve1[i]
            p1_later = curve2[i]
            # positive number if p1_later gets closer to p2
            dist = get_direction_and_distance(p1, p2,
                                              p1, p1_later,
                                              perpendicular=perpendicular)
            total_displacement += dist
    return total_displacement

def compute_length(blob_id, **kwargs):
    times, spines, _ = get_data(blob_id, data_type='spine', **kwargs)
    na_values = ['', -1, 'NA', 'NaN', None, []]
    ltimes, lengths = [], []
    for t, spine in izip(times, spines):        
        if len(spine) != 0 or spine not in na_values:
            ltimes.append(t)
            lengths.append(length_of_spine(spine))
    return times, lengths

def compute_speed_along(blob_id, **kwargs):
    times, spines, _ = get_data(blob_id, data_type='spine', **kwargs)
    stimes, speeds = [], []
    for i in range(len(times)):
        if i + 1 < len(spines) and len(spines[i]) > 0 and len(spines[i + 1]) > 0:
            speed = speed_along_spine(spines[i], spines[i + 1],
                                      times[i + 1], times[i])
            stimes.append(time[i])
            speeds.append(speed)
    return stimes, speeds

def compute_speed_perp(blob_id, **kwargs):
    times, spines, _ = get_data(blob_id, data_type='spine', **kwargs)
    stimes, speeds = [], []
    perpendicular = True
    for i in range(len(times)):
        if i + 1 < len(spines) and len(spines[i]) > 0 and len(spines[i + 1]) > 0:
            speed = speed_along_spine(spines[i], spines[i + 1],
                                      times[i + 1], times[i], perpendicular=True)
            stimes.append(t)
            speeds.append(speed)
    return stimes, speeds

def compute_curvature(blob_id, **kwargs):
    times, spines, _ = get_data(blob_id, data_type='spine', **kwargs)
    ctimes, curvatures = [], []
    for t, spine in izip(times, spines):
        if len(spine) > 0:
            curve = curvature_of_spine(spine_list[i], points, scaling_factor)
            ctimes.append(t)
            curvatures.append(curve)
    return ctimes, curvatures

def curvature_of_spine(spine, points='all'):
    def compute_curvature_from_three_points(p, q, t):
        """
        Solution from: http://mathworld.wolfram.com/SSSTheorem.html
        p,q and t are tuples (x,y)
        """
        a = euclidean(p, q)
        b = euclidean(t, q)
        c = euclidean(p, t)
        if a == 0 or b == 0 or c == 0:
            return 0.
        s = 0.5 * (a + b + c)
        # preventing round off error from breaking code
        if (s * (s - a) * (s - b) * (s - c)) < 0:
            K = 0.
        else:
            K = math.sqrt(s * (s - a) * (s - b) * (s - c))
        return K / (a * b * c) * 4.
    if points == 'all':
        points = range(len(spine))
    curvatures = []
    for i in points:
        if 1 < i < len(spine) - 1:
            curvature = compute_curvature_from_three_points(spine[i - 1], spine[i], spine[i + 1])
            curvatures.append(curvature)
    return np.mean(curvatures)
