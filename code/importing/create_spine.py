#!/usr/bin/env python

'''
Filename: create_spine.py
Description: find_ex_ids_to_update function is create_spine_from_outline
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import numpy as np
from itertools import izip

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__)) 
PROJECT_DIR = os.path.abspath(HERE + '/../../')
SHARED_DIR = os.path.abspath(PROJECT_DIR + 'code/shared/')
sys.path.append(PROJECT_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from skeletonize_outline import compute_skeleton_from_outline
from Encoding.decode_outline import decode_outline
from filtering.equally_space import equally_space
from filtering.filter_utilities import savitzky_golay
from settings.local import SMOOTHING
from wio.file_manager import get_timeseries, insert_data_into_db, write_tmp_file

# set defaults from settings file
DEFAULT_ORDER = SMOOTHING['spine_poly_order']
DEFAULT_WINDOW = SMOOTHING['spine_window_size']

def create_spine_from_outline(blob_id, store_tmp=True, verbose=False, **kwargs):
    '''
    pulls encoded outline documents from the database calculates a centerline
    called 'spine', smoothes it using a polynomial-smoothing technique, 
    equally spaces 50 points along the length of the centerline, and inserts
    the centerline data back into the database.

    inputs: blob_id

    '''
    # if temp data is cached, use that instead of querying database
    times, encoded_outlines = get_timeseries(blob_id=blob_id, data_type='encoded_outline')
    outlines = []
    spines = []
    flagged_timepoints = []
    num_short_spines = 0
    for t, encoded_outline in izip(times, encoded_outlines):
        outline = decode_outline(encoded_outline)
        # if error in decoding outline
        if len(outline) == 0:
            outlines.append([])
            spines.append([])
            flagged_timepoints.append(t)
            continue
        '''
        # do I really need such a strange test?
        for pt in outline:
            if type(pt) != tuple:
                print outline, 'warning: outline points not all tuples'
            if len(pt) != 2:
                print outline, 'warning: outline points not all in pairs'
            try:
                int(pt[0]) + int(pt[1])
            except Exception as e:
                print e, '\n', outline, 'outline int test failed'
        '''
        outlines.append(outline)
        try:
            spine = compute_skeleton_from_outline(outline)
            spines.append(spine)
            if len(spine) == 0:
                num_short_spines += 1
        except Exception as e:
            print e
            print 'Warning: skeleton reconstruction failed for time {t}'.format(t=t)
            spines.append([])
            flagged_timepoints.append(t)
    print '\tN flags during spine creation: {N}'.format(N=len(flagged_timepoints))
    print '\tN spines too short: {N}'.format(N=num_short_spines)
    # equally spaces points and removes reversals of head and tail in the worm spines
    treated_spines = treat_spine(times, spines)
    #show_worm_video(treated_spine_timedict)
    # insert it back into the database
    data_type = 'spine_rough'
    if store_tmp:        
        data ={'time':times, 'data':treated_spines}
        write_tmp_file(data=data, blob_id=blob_id, data_type=data_type)
    return times, treated_spines, flagged_timepoints

def show_worm_video(spine_timedict):
    '''
    quick dirty way of sequentially looking at images in a spine timedict.
    '''
    import pylab as pl

    times = sorted([(float(t.replace('?', '.')), t) for t in spine_timedict])
    pl.ion()
    for tf, t in times[:]:
        spine = spine_timedict[t]

        sx = [v[0] for v in spine] 
        sy = [v[1] for v in spine]
        pl.plot(sx, sy, color='blue')

        center_x, center_y = spine[len(spine) / 2]
        window_size = 2
        pl.xlim([int(center_x) - window_size, int(center_x) + window_size])
        pl.ylim([int(center_y) - window_size, int(center_y) + window_size])
        pl.raw()
        pl.clf()

def smooth_and_space_xy_points(points, poly_order=DEFAULT_ORDER, window_size=DEFAULT_WINDOW, point_num=50):
    xs, ys = zip(*points)
    filtered_xs = list(savitzky_golay(np.array(xs), window_size=window_size, order=poly_order))
    filtered_ys = list(savitzky_golay(np.array(ys), window_size=window_size, order=poly_order))
    return equally_space(zip(filtered_xs, filtered_ys), points=point_num)


def treat_spine(times, spines, poly_order=DEFAULT_ORDER, window_size=DEFAULT_WINDOW, verbose=True):
    """
    this function returns a recalculated spine_timedict that has been polynomially smoothed and now contains 50 points
    along it's centerline. Each spines now faces in the same direction as the spine one timestep earlier.
    
    order of operations:
    1. smooth the existing points using 
    2. equally space and standardize number of points to 50.
    3. all spines are made to consitantly face in the same direction by reversing the order of some of the spines.

    if the spine has len > than windowsize: smooth, equally space 50 points, and
    switch head-tails so all spines are consistent. returns new dictionary.
    """


    #times = sorted([(float(t.replace('?', '.')), t) for t in spine_timedict])
    # need seperate list of keys in correct order because floats sort differently than strings.
    # t_keys = []

    # 1. smooth pixelated spine matrix
    goodcount = 0
    badcount = 0
    treated_spines = []
    for t, spine in izip(times, spines):
        #spine = spine_timedict[t_key]
        #print len(spine), 'num points in raw spine'

        if len(spine) > 1.5 * window_size:
            xs, ys = zip(*spine)
            filtered_xs = list(savitzky_golay(np.array(xs), window_size, poly_order))
            filtered_ys = list(savitzky_golay(np.array(ys), window_size, poly_order))
            treated_spines.append(zip(filtered_xs, filtered_ys))
            #t_keys.append(t_key)
            goodcount += 1
        else:
            badcount += 1
            treated_spines.append([])
            #print 'Warning: len spine smaller than polynomial smoothing window:', len(spine), t_key
    if verbose:            
        N = len(treated_spines)
        print '\tgood: {g} | bad: {b} | total: {N}'.format(g=goodcount, b=badcount, N=N)

    #ion()
    #2. equally space and reverse points if backwards
    
    treated_spines = map(lambda x: equally_space(x, points=50), treated_spines)

    #standardized_spines = [treated_spines[0]]
    for i in range(len(treated_spines)):
        if i > 0:
            last_spine = treated_spines[i - 1]
            this_spine = treated_spines[i]
            if this_spine and last_spine:
                assert len(last_spine) == len(this_spine), 'spines unequal len: %i -- %i ' % (len(last_spine),
                                                                                              len(this_spine))
            #print 'spines are len: %i -- %i ' %(len(last_spine), len(this_spine))
            # this makes all spines consistently facing same direction
                final_spine, reversed_flag = reverse_points_if_backwards(last_spine, this_spine)
            else:
                # if both not present, then just use the current spine
                final_spine = this_spine
            #standardized_spines.append(final_spine)
            treated_spines[i] = final_spine
            #x, y = zip(*final_spine)
            #plot(x,y, ls='',marker='o', alpha=0.5)

            #draw()
            #clf()

    # this piece of code is checking if the spines are ok after being reversed once
    reversed_flag_total=False
    for i in range(len(treated_spines)):
        if i>0:
            last_spine = treated_spines[i-1]
            this_spine = treated_spines[i]
            if this_spine and last_spine:
                final_spine, reversed_flag = reverse_points_if_backwards(last_spine, this_spine)
                if reversed_flag:
                    print 'it happened again!!!!!!!!!', i
                    print last_spine
                    print this_spine
                    reversed_flag_total=True
            else:
                #print i, ' was ok'
                pass
            #treated_spine_timedict[t_key] = final_spine

    print '\trough spines aligned: {b}'.format(b= not reversed_flag_total)
    return treated_spines


def reverse_points_if_backwards(xy, xy_next):
    """
    This function aligns xy_next so that it is in the same direction as xy.
    Nothing occurs if they are already aligned

    inputs:
    xy, xy_next - list of tuples [(x1, y1), (x2, y2) ...]
    xy and xy_next are seperated by one timestep.
    the function returns the reversed spine and a flag to see it was reversed
    """

    x, y = zip(*xy)
    xnext, ynext = zip(*xy_next)
    xnext_rev = xnext[::-1]
    ynext_rev = ynext[::-1]

    distance_original = 0.
    distance_rev = 0.
    for k in range(len(x)):
        distance_original += ((x[k] - xnext[k]) ** 2 + (y[k] - ynext[k]) ** 2)
        distance_rev += (x[k] - xnext_rev[k]) ** 2 + (y[k] - ynext_rev[k]) ** 2
        if (distance_original > distance_rev):
            #print "reversed", index, distance_rev, distance_original
            newxy = zip(xnext_rev, ynext_rev)
            return (newxy, True)
        else:
            #print "ok", index
            return (xy_next, False)


if __name__ == "__main__":
    blob_id = '20121124_181927_00197'
    blob_id = '20130319_150235_00426'
    create_spine_from_outline(blob_id)
