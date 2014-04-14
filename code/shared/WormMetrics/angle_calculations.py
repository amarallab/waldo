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

rad_to_deg_factor = 180.0 / np.pi

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

def angle_change_for_xy(x, y, allow_negatives=True, units='deg'):
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
    print 'nan counts', nan_counts
    return angles

# Currently nonfunctional....
def angle_change_for_xy2(x, y):
    dx, dy =np.diff(x), np.diff(y)
    angles = np.arctan2(dy, dx) * rad_to_deg_factor
    print angles
    print 'done'
    #unit_vectors = [v / np.linalg.norm(v) for v in vectors]
    '''
    angles = []
    for v1, v2 in zip(unit_vectors[:-1], unit_vectors[1:]):
        angle = np.arccos(np.dot(v1, v2)) * rad_to_deg_factor
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
    
