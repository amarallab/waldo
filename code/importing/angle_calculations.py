#!/usr/bin/env python

'''
Filename: angle_calculations.py
Description: functions to calculate angles
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

import numpy as np
import math

def angle_change_for_xy(x, y, allow_negatives=True):
    vectors = np.array(zip(np.diff(x), np.diff(y)))
    unit_vectors = [v / np.linalg.norm(v) for v in vectors]    
    angles = []        
    for v1, v2 in zip(unit_vectors[:-1], unit_vectors[1:]):
        angle = np.arccos(np.dot(v1, v2)) * 180.0 / np.pi
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
    return angles

def angle_change_for_xy2(x, y):
    dx, dy =np.diff(x), np.diff(y)
    angles = np.arctan2(dy, dx) * 180 / np.pi
    print angles
    print 'done'
    #unit_vectors = [v / np.linalg.norm(v) for v in vectors]
    '''
    angles = []
    for v1, v2 in zip(unit_vectors[:-1], unit_vectors[1:]):
        angle = np.arccos(np.dot(v1, v2)) * 180.0 / np.pi
        print v1, v2, angle, np.cross(v1, v2)
        #if np.dot(v1, v2) < 0.0:
        #    angle = -1 * angle
            
        if np.isnan(angle):
            if (v1 == v2).all():
                angle ==  0.0
            else:
                angle ==  180        
        angles.append(angle)
    '''
    return angles


def angle_change_over_distance(blob_id, store_tmp=True, step=0.1, **kwargs):
    times, xy = get_timeseries(blob_id, data_type='xy', **kwargs)
    times = np.array(times)
    x, y = map(np.array, zip(*xy))
    t, x, y = equally_space_xy_for_stepsize(times, x, y, step=step)
    angles = angle_change_for_xy(x, y)        
    
