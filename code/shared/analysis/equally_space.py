#!/usr/bin/env python

'''
Filename: equally_space_old.py
Description:
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

import os
import sys
import numpy as np
import scipy.interpolate as interpolate
from itertools import izip
import random
import math

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../shared/')
PROJECT_DIR = os.path.abspath(HERE + '/../../')
assert os.path.exists(SHARED_DIR), 'shared code directory not found'
sys.path.append(SHARED_DIR)

# nonstandard imports
from filtering.filter_utilities import savitzky_golay

NUM_POINTS = 50

def euclidean(list1, list2):
    ''' calculates euclidean distance between numerical values in two lists'''
    return math.sqrt(sum((x - y) ** 2 for x, y in izip(list1, list2)))

def equally_spaced_tenth_second_times(start=0.0, end=60.0):
    starti, endi = round(start, ndigits=1), round(end, ndigits=1)
    N = int((endi - starti) / 0.1 )
    # include this check to be absolutely certain all new
    # times fall inside start and end times
    times = []
    for t in [i/10. + starti for i in xrange(N+1)]:
        if start <= t <= end:
            times.append(t)
    return times

def equally_spaced_times(start=0.0, end=60.0, dt=1.0, ndigits=0):

    starti, endi = round(start, ndigits=ndigits), round(end, ndigits=ndigits)
    N = int((endi - starti) / dt)
    # include this check to be absolutely certain all new
    # times fall inside start and end times
    times = []
    for t in [i/10. + starti for i in xrange(N+1)]:
        if start <= t <= end:
            times.append(t)
    return times

def equally_space_N_xy_points(x, y, N=50, kind='linear'):
    # get distance traveled between each point
    N = len(x)
    tot_dist= 0
    #dists = [tot_dist]
    #last_pt = [x[0], y[0]]
    #for pt in izip(x[1:], y[1:]):
    dists = []
    last_pt = [x[0], y[0]]
    for pt in izip(x, y):
        tot_dist += euclidean(pt, last_pt)
        dists.append(tot_dist)
        last_pt = pt
    # scale distances and use like unevenly spaced timepoints
    dists = [(N-1) * d/tot_dist for d in dists]
    interp_x = interpolate.interp1d(dists, x, kind=kind)
    interp_y = interpolate.interp1d(dists, y, kind=kind)
    # get new xy values in which equal distance is traveled between.
    new_x = interp_x(range(N))
    new_y = interp_y(range(N))
    return new_x, new_y

def equally_space_xy_for_stepsize(t, x, y, step=0.5, kind='linear', n_interp_pts=50):
    """ returns a x and a y list with points that are equal distance
    from one another in space. The number of points depends on step size.

    Arguments:
    - `x`, 'y': (list or d1 array) of x, y coords.
    - `step`: (float) euclidean distance between points
    - `kind`: (str) type of interpolation used to calculate values between points.
        recomended options: 'linear', 'cubic'
        non-recomended options: 'nearest', 'zero', 'slinear', 'quadratic,
    - `n_interp_pts`: (int) how many points are interpolated
    """
    # create interpolation objects
    N = len(x)
    interp_x = interpolate.interp1d(range(N), x, kind=kind)
    interp_y = interpolate.interp1d(range(N), y, kind=kind)
    interp_t = interpolate.interp1d(range(N), t, kind=kind)
    # create an excess of xy points using interpolation
    N2 = int((N - 1) * n_interp_pts)
    pseudo_t = [i/float(n_interp_pts) for i in xrange(N2)]
    t, x, y = interp_t(pseudo_t), interp_x(pseudo_t), interp_y(pseudo_t)
    # initialize new lists with first point
    new_t, new_x, new_y = [t[0]], [x[0]], [y[0]]
    # only keep xy points if they are at further from last point by step dist.
    for tp, xp, yp in izip(t, x, y):
        if step <= euclidean([new_x[-1], new_y[-1]], [xp, yp]):
            new_t.append(tp)
            new_x.append(xp)
            new_y.append(yp)
    #print 'from t to s we go from {N} to {n} points'.format(N=N, n=len(new_x))
    return new_t, new_x, new_y

def create_spine_matricies(times, spines):
    # find the longest spine, if no spines have lenght, return False
    N_pts = max([len(s) for s in spines])
    if not N_pts:
        return times, False, False
    # only keep spines that match longest length. otherwise remove.
    good_count, bad_count = 0, 0
    g_times, g_spines = [], []
    for t, spine in izip(times, spines):
        if len(spine) == N_pts:
            g_times.append(t)
            g_spines.append(spine)
            good_count += 1
        else:
            bad_count += 1
    print '\tcreating spine matricies. good: {g} | bad: {b}'.format(g=good_count, b=bad_count)
    # initialize matrix sizes to number of good spines, and longest lenght spine
    N = len(g_spines)
    x_matrix = np.zeros((N, N_pts), dtype=float)
    y_matrix = np.zeros((N, N_pts), dtype=float)
    #print 'x dimensions', len(x_matrix), len(x_matrix[0])
    #print 'y dimensions', len(y_matrix), len(y_matrix[0])
    for i,pts in enumerate(g_spines):
        x_matrix[i], y_matrix[i] = zip(*pts)
    return g_times, x_matrix, y_matrix

def equally_space_matricies_times(eq_times, orig_times, x_mat, y_mat):
    kind = 'linear'
    N_cols = len(x_mat[0])
    # make absolutely sure that intepolated times are inside origional timerange
    if eq_times[0] < orig_times[0]:
        eq_times = eq_times[1:]
    if eq_times[-1] > orig_times[-1]:
        eq_times = eq_times[:-1]
    # initialize empty arrays for x,y
    N_times = len(eq_times)
    x_new = np.zeros((N_times, N_cols), dtype=float)
    y_new = np.zeros((N_times, N_cols), dtype=float)
    #print 'x dimensions', len(x_mat), len(x_mat[0])
    #print 'y dimensions', len(y_mat), len(y_mat[0])
    #print 't', len(orig_times)
    for col in range(N_cols):
        # interpolate new x, y values
        interp_x = interpolate.interp1d(orig_times, x_mat[:,col], kind=kind)
        interp_y = interpolate.interp1d(orig_times, y_mat[:,col], kind=kind)
        x_new[:,col] = interp_x(eq_times)
        y_new[:,col] = interp_y(eq_times)
    return x_new, y_new

def smooth_matricies_cols(x_mat, y_mat, window, order):
    for col in xrange(len(x_mat[0])):
        x_mat[:,col] = savitzky_golay(x_mat[:,col], window_size=window, order=order)
        y_mat[:,col] = savitzky_golay(y_mat[:,col], window_size=window, order=order)
    return x_mat, y_mat

def smooth_matricies_rows(x_mat, y_mat, window, order):
    for row in xrange(len(x_mat)):
        x_mat[row] = savitzky_golay(x_mat[row], window_size=window, order=order)
        y_mat[row] = savitzky_golay(y_mat[row], window_size=window, order=order)
    return x_mat, y_mat

def set_matrix_orientation(x_mat, y_mat, verbose=True):
    if len(x_mat) <= 2:
        return x_mat, y_mat
    forward_counts, backward_counts, dist_forward = 0, 0, 0.
    N_rows, N_cols = len(x_mat), len(x_mat[0])
    for row in xrange(N_rows-1):
        dx = np.diff(x_mat[row:row+2], axis=0)[0]
        dy = np.diff(y_mat[row:row+2], axis=0)[0]
        dist = np.sum(np.sqrt(dx**2 +dy**2))
        if dist > 0.0:
            forward_counts += 1
        else:
            backward_counts += 1
        dist_forward += dist
    if verbose:
        h, t, d = forward_counts, backward_counts, round(dist_forward, ndigits=1)
        print '\tdirection head: {h} | tail: {t} | dist: {d} pix'.format(h=h, t=t, d=d)
    if dist_forward < 0.0:
        x_mat, y_mat = x_mat[:,::-1], y_mat[:, ::-1]
    return x_mat, y_mat

def equally_space_matrix_distances(x_mat, y_mat):
    kind = 'linear'
    # get distance traveled between each point
    N_rows, N_cols = len(x_mat), len(x_mat[0])
    for row in xrange(N_rows):
        # intialize distances for row and save first point.
        tot_dist, dists = 0, []
        pt = [x_mat[row,0], y_mat[row, 0]]
        for col in xrange(N_cols):
            # find xy, increment points, save distances
            x, y = x_mat[row, col], y_mat[row, col]
            pt, last_pt = [x, y], pt
            tot_dist += euclidean(pt, last_pt)
            dists.append(tot_dist)
        # scale distances to go from 0 to 1
        dists = [(d / tot_dist) for d in dists]
        interp_x = interpolate.interp1d(dists, x_mat[row], kind=kind)
        interp_y = interpolate.interp1d(dists, y_mat[row], kind=kind)
        # get new xy values in which equal distance is traveled between.
        eq_d = np.linspace(0, 1, num=N_cols)
        x_mat[row] = interp_x(eq_d)
        y_mat[row] = interp_y(eq_d)
    return x_mat, y_mat

def spine_matricies_to_points(x_mat, y_mat):
    spines = []
    N_rows, N_cols = len(x_mat), len(x_mat[0])
    x_new = np.zeros((N_rows, N_cols), dtype=float)
    y_new = np.zeros((N_rows, N_cols), dtype=float)
    for row, _ in enumerate(x_mat):
        spine = []
        for col, _ in enumerate(x_mat[0]):
            x, y = x_mat[row, col], y_mat[row, col]
            spine.append((float(x),float(y)))
        spines.append(spine)
    return spines

def dists(xs, ys):
    """
    """
    d = []
    last_x, last_y = None, None
    for x, y in izip(xs, ys):
        if last_x:
            d.append(euclidean([x, y], [last_x, last_y]))
        last_x, last_y = x, y
    print 'mean: {m}, std: {s}'.format(m=np.mean(d), s=np.std(d))
    #print 'x range', xs[0], xs[-1]
    #print 'y range', ys[0], ys[-1]

if __name__ == '__main__':

    # kind options
    # 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'

    x = [(i + random.uniform(-0.5, 0.5))for i in xrange(10)]
    y = [(i + random.uniform(-0.5, 0.5))for i in xrange(10)]
    x1,y1 = equally_space_N_xy_points(x, y, N=50)
    '''
    dists(x, y)
    step = 0.1
    kind = 'linear'
    print kind
    x1, y1 = xy_time_to_distance(x, y, step=step, kind=kind)
    dists(x1, y1)
    print 'x, y leftovers: {x}, {y}'.format(x=x[-1]-x1[-1],
                                                    y=y[-1]-y1[-1])
    kind ='cubic'
    print kind
    x2, y2 = xy_time_to_distance(x, y, step=step, kind=kind)
    dists(x2, y2)
    print 'x, y leftovers: {x}, {y}'.format(x=x[-1]-x2[-1], y=y[-1]-y2[-1])
    '''
