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

RAD_TO_DEG = 180.0 / np.pi

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
    speeds = txy_to_speeds(t=times, x=x, y=y)
    return times, speeds

def compute_angle_change(blob_id, **kwargs):
    times, xy = get_timeseries(blob_id, data_type='xy', **kwargs)
    if len(times) ==0 or len(xy) ==0:
        return [], []
    x, y = zip(*xy)
    angle_change = angle_change_for_xy(x=x, y=y)
    # not make angle change over time.
    angle_change_dt = angle_change[1:-1] * np.diff(np.array(t))        
    return times, angle_change_dt

def txy_to_speeds(t, x, y):
    dt = np.diff(np.array(t))
    dx = np.diff(np.array(x))
    dy = np.diff(np.array(y))
    # to guard against division by zero
    for i, t in enumerate(dt):
        if t < 0.0000001:
            dt[i] = 0.0000001
    speeds = np.sqrt(dx**2 + dy**2) / dt
    return list(speeds) + [np.nan]

def rescale_radians(rad, max_iter=10000):    
    for i in xrange(max_iter):
        rad -= 2 * np.pi * int(rad / (2*np.pi))
        if rad < -np.pi:
            rad += 2 * np.pi
        elif rad > np.pi:
            rad -= 2 * np.pi
        if -np.pi < rad < np.pi:        
            return rad
    else:
        print 'Error: rescaling failed for: {r}'.format(r=rad)
        return False        

def angle_change_for_xy(x, y, allow_negatives=True, units='rad'):
    vectors = np.array(zip(np.diff(x), np.diff(y)))
    unit_vectors = [v / np.linalg.norm(v) for v in vectors]    
    angles = []
    nan_counts = 0
    for v1, v2 in zip(unit_vectors[:-1], unit_vectors[1:]):
        if np.isnan(v1[0]):
            nan_counts += 1
        angle = np.arccos(np.dot(v1, v2))
        if units == 'deg':
            angle = angle * rad_to_deg_factor
        # if allow negatives, see if direction of 3d cross product + or -
        if allow_negatives:
            # make sure vectors are 3D
            va = np.array([v1[0], v1[1], 0])
            vb = np.array([v2[0], v2[1], 0])            
            cross = np.cross(va, vb)[2]
            # take sign of angle change
            if cross != 0:
                angle = math.copysign(1.0, cross) * angle
        # catch any right angle exceptions
        if np.isnan(angle):
            if (v1 == v2).all():
                angle ==  0.0
            else:
                angle ==  180        
        angles.append(angle)

    #print 'nan counts', nan_counts
    return [np.nan] + list(angles) + [np.nan]
        
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
