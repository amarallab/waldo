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
import numpy as np
from itertools import izip
import scipy.stats as stats

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(HERE + '/../../')
CODE_DIR = os.path.abspath(HERE + '/../')
SHARED_DIR = CODE_DIR + 'shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

from GeometricCalculations.distance import euclidean
from wio.file_manager import get_blob_ids, get_timeseries
from importing.flags_and_breaks import consolidate_flags

# TODO? move remove flag functionality further upstream?
def pull_basic_data_type(blob_id, data_type, remove_flags=True, **kwargs):
    ''' default option to remove flagged timepoints!'''
    times, data = get_timeseries(blob_id, data_type=data_type, **kwargs)

    # make sure things don't crash
    if type(times) == None or type(data) == None:
        return [], []
    if len(times) == 0 or len(data) == 0:
        return [], []

    if remove_flags:
        times_f, all_flags = get_timeseries(blob_id, data_type='flags', **kwargs)
        flags = consolidate_flags(all_flags)
        #print data_type
        #print type(times), type(data), type(flags)
        #print len(times), len(data), len(flags)
        unflagged_timeseries = [(t, s) for (t, s, f) in izip(times, data, flags) if f]
        if type(unflagged_timeseries) == None or len(unflagged_timeseries) == 0:
            return [], []
        times, data = zip(*unflagged_timeseries)
    return times, data

# todo
def space_basic_data_type_linearly(times, values):
    return times, values


def quantiles_for_data(data, quantiles=range(10,91, 10)):
    data = [d for d in data if not np.isnan(d)]
    return [stats.scoreatpercentile(data, q) for q in quantiles]

def compute_size(blob_id, **kwargs):    
    # size is a body shape property. if we've flagged the shape, don't measure size.
    return pull_basic_data_type(blob_id, data_type='size',
                                remove_flags=True, **kwargs)

def compute_width(blob_id, **kwargs):
    # size is a body shape property. if we've flagged the shape, don't measure size.
    return pull_basic_data_type(blob_id, data_type='width50', 
                                remove_flags=True, **kwargs)

def compute_centroid_speed(blob_id, **kwargs):
    # speed is not a body shape.
    times, xy = get_timeseries(blob_id, data_type='xy', **kwargs)
    if len(times) ==0 or len(xy) ==0:
        return [], []
    x, y = zip(*xy)
    dt = np.diff(np.array(times))
    dx = np.diff(np.array(x))
    dy = np.diff(np.array(y))
    # to guard against division by zero
    for i, t in enumerate(dt):
        if t < 0.0000001:
            dt[i] = 0.0000001
    speeds = np.sqrt(dx**2 + dy**2) / dt
    return times[1:], speeds
    
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

def displacement_along_curve(curve1, curve2, perpendicular=False):
    x1, y1 = map(np.array, map(list, zip(*curve1)))
    x2, y2 = map(np.array, map(list, zip(*curve2)))
    # change in position along curve
    dx1, dy1 = np.diff(x1), np.diff(y1)
    N = len(dx1)
    # change in position over time
    dx = (x1 - x2)[:N]
    dy = (y1 - y2)[:N]
    # dot product of different changes in position
    dot = dx*dx1 + dy*dy1
    norm = np.sqrt(dx1**2 + dy1**2)
    if not perpendicular:
        return sum(dot / norm)
    else:
        norm2 = np.sqrt(dx**2 + dy**2)
        cosine_ = dot /norm / norm2
        sin_ = np.sqrt(1. - cosine_ ** 2)
        # to guard against round off errors        
        for i, c in enumerate(cosine_):            
            if (1. - c ** 2) < 0:
                sin_[i] = 0.
        perp_projection = np.fabs(norm2 * sin_)
        return sum(perp_projection)

# andrea's old code.
'''
def displacement_along_curve2(curve1, curve2, perpendicular=False, points='all'):
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
    
def get_direction_and_distance(p1, p2, m1, m2, perpendicular=False):
    """
        p1, p2, m1, and m2 are all tuples (x,y)
        the function computes the projection of the 
        vector m2-m1 along the vector p2-p1
        if the projection is positive it means 
        that m2 is getting closer to p2
        the number tells how much it got closer
        N.B. the perpendicular_projection is always
        positive: we want to know how much is moving
        perpendicularly to the curve
        but without caring if it is left ot right
    """

    vector_p2_p1 = (p2[0] - p1[0], p2[1] - p1[1])
    vector_m2_m1 = (m2[0] - m1[0], m2[1] - m1[1])

    norm_p2_p1 = math.sqrt(vector_p2_p1[0] ** 2 + vector_p2_p1[1] ** 2)
    norm_m2_m1 = math.sqrt(vector_m2_m1[0] ** 2 + vector_m2_m1[1] ** 2)
    if norm_p2_p1 == 0 or norm_m2_m1 == 0:
        return 0.

    dot_product = vector_p2_p1[0] * vector_m2_m1[0] + \
                  vector_p2_p1[1] * vector_m2_m1[1]

    if perpendicular:
    #use sin
        norm_m2_m1_square = vector_m2_m1[0] ** 2 + vector_m2_m1[1] ** 2
        cosine_ = dot_product / (norm_p2_p1) / math.sqrt(norm_m2_m1_square)
        # to guard against round off errors
        if (1. - cosine_ ** 2) < 0:
            sin_ = 0.
        else:
            sin_ = math.sqrt(1. - cosine_ ** 2)
        perpendicular_projection = math.fabs(math.sqrt(norm_m2_m1_square) * sin_)

        # this can be skipped setting check_=False
        check_ = False
        if check_:
            # check with pitagora's theorem
            projection = dot_product / norm_p2_p1
            perpendicular_projection2 = math.sqrt(norm_m2_m1_square - projection ** 2)
            assert math.fabs(perpendicular_projection - perpendicular_projection2) < 1e-8
            # check with line intersections
            if vector_p2_p1[0] == 0:
                perpendicular_projection3 = math.fabs(vector_m2_m1[0])
            else:
                ratio = vector_p2_p1[1] / vector_p2_p1[0]
                a = 1. / math.sqrt(1 + ratio ** 2)
                perpendicular_projection3 = vector_m2_m1[0] * (-ratio) * a + \
                                            vector_m2_m1[1] * a
                perpendicular_projection3 = math.fabs(perpendicular_projection3)
            assert math.fabs(perpendicular_projection - perpendicular_projection3) < 1e-8

        return perpendicular_projection
    else:
        return -1 * dot_product / norm_p2_p1
'''

def compute_length(blob_id, **kwargs):
    times, spines = get_timeseries(blob_id, data_type='spine', **kwargs)
    if spines==None or len(spines) ==0:
        return [], []
    na_values = ['', -1, 'NA', 'NaN', None, []]
    ltimes, lengths = [], []
    for t, spine in izip(times, spines):        
        if len(spine) != 0 or spine not in na_values:
            ltimes.append(t)
            lengths.append(length_of_spine(spine))
    return times, lengths

def compute_speed_along(blob_id, **kwargs):
    times, spines = get_timeseries(blob_id, data_type='spine', **kwargs)
    if spines==None or len(spines) ==0:
        return [], []
    stimes, speeds = [], []
    for i in range(len(times)):
        if i + 1 < len(spines) and len(spines[i]) > 0 and len(spines[i + 1]) > 0:
            speed = speed_along_spine(spines[i], spines[i + 1],
                                      times[i + 1], times[i])
            stimes.append(time[i])
            speeds.append(speed)
    return stimes, speeds

def compute_speed_perp(blob_id, **kwargs):
    times, spines = get_timeseries(blob_id, data_type='spine', **kwargs)
    if spines==None or len(spines) ==0:
        return [], []
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
    times, spines = get_timeseries(blob_id, data_type='spine', **kwargs)
    if spines==None or len(spines) ==0:
        return [], []
    ctimes, curvatures = [], []
    for t, spine in izip(times, spines):
        if len(spine) > 0:
            curve = curvature_of_spine(spine)
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
