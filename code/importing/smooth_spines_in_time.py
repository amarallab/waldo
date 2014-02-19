#!/usr/bin/env python

'''
Filename: smooth_spines_in_time.py
Description: The functions in this script take consecutive time-points (or 'good regions') of previously calculated
spines (1) smooth each region in time and space and (2) order them such that the 'head' of the worms is set for
each region to be the direction of most motion.

'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
from itertools import izip
import pylab as pl
#from scipy.spatial.distance import euclidean

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__)) 
PROJECT_DIRECTORY = os.path.abspath(HERE + '/../../')
SHARED_DIRECTORY = PROJECT_DIRECTORY + '/code/shared/'
assert os.path.exists(PROJECT_DIRECTORY), 'project directory not found'
assert os.path.exists(SHARED_DIRECTORY), 'shared code directory not found'
sys.path.append(SHARED_DIRECTORY)

# nonstandard imports
from GeometricCalculations import compute_displacement_along_curve
from GeometricCalculations.distance import euclidean
from wio.file_manager import get_data, write_tmp_file, store_data_in_db
from flags_and_breaks import good_segments_from_data, get_flagged_times
from create_spine import smooth_and_space_xy_points
from equally_space import *
from settings.local import SMOOTHING 

# set defaults from settings file
TIME_POLY_ORDER = SMOOTHING['time_poly_order']
TIME_WINDOW_SIZE = SMOOTHING['time_window_size']
SPINE_POLY_ORDER = SMOOTHING['spine_poly_order']
SPINE_WINDOW_SIZE = SMOOTHING['spine_window_size']
T_STEP = SMOOTHING['time_step']
N_POINTS = SMOOTHING['N_points']

def smooth_good_regions_repeatedly(blob_id, repeated_smoothings=5,
                                   spine_poly_order=SPINE_POLY_ORDER,
                                   spine_running_window_size=SPINE_WINDOW_SIZE,
                                   time_poly_order=TIME_POLY_ORDER,
                                   time_running_window_size=TIME_WINDOW_SIZE,
                                   store_in_db=False, store_tmp=True,
                                   time_step=T_STEP, **kwargs):
    """
    Returns a time-dict of spines that have been (1.) repeatedly smoothed in space and then in time
    (2.) oriented such that the majority of movement goes towards the front of the spine.

    :param blob_id: blob_id used to
    :param repeated_smoothings: number of times smoothing process should be repeated. (int)
    :param spine_poly_order:
    :param spine_running_window_size:
    :param time_poly_order:
    :param time_running_window_size:
    :param insert: toggle if resulting data should be inserted into database.
    :return: timedict of smooothed spines.
    """
    # get data into proper form
    break_list, _ = get_data(blob_id, data_type='breaks',
                             split_time_and_data=False, **kwargs)
    times, spines, db_doc = get_data(blob_id, data_type='treated_spine', **kwargs)
    flagged_times = get_flagged_times(blob_id)
    good_regions = good_segments_from_data(break_list, times=times, data=spines,
                                           flagged_times=flagged_times)
                                           
                                           
    # initialize buckets
    smoothed_times, smoothed_spines = [], []
    # each region is smoothed independently
    for region in good_regions:
        # if region is too small, it is not worth smoothing
        if len(region) < spine_running_window_size:
            continue
        # transform spine point format into matrix format
        times, spines = zip(*region)
        x_matrix, y_matrix = create_spine_matricies(spines)
        # smooth once in both directions and make sure points are equally spaced along spine
        x_matrix, y_matrix = smooth_matricies_cols(x_matrix, y_matrix, window=time_running_window_size, order=time_poly_order)
        x_matrix, y_matrix = smooth_matricies_rows(x_matrix, y_matrix, window=spine_running_window_size, order=spine_poly_order)
        x_matrix, y_matrix = equally_space_matrix_distances(x_matrix, y_matrix)
        # interpolate missing times 
        eq_times = equally_spaced_tenth_second_times(start=times[0], end=times[-1])
        x_matrix, y_matrix = equally_space_matricies_times(eq_times, times, x_matrix, y_matrix)
        # now that times have been set, smooth + space spines repeatedly

        for i in range(repeated_smoothings):
            x_matrix, y_matrix = smooth_matricies_cols(x_matrix, y_matrix, window=time_running_window_size, order=time_poly_order)
            x_matrix, y_matrix = smooth_matricies_rows(x_matrix, y_matrix, window=spine_running_window_size, order=spine_poly_order)
            x_matrix, y_matrix = equally_space_matrix_distances(x_matrix, y_matrix)
        # check if head is correct and reverse row orientation if not
        x_matrix, y_matrix = set_matrix_orientation(x_matrix, y_matrix)
        # transform spine matrix format into point format
        spines = spine_matricies_to_points(x_matrix, y_matrix)
        # add an empty string one space after end of region.
        #times.append(str(float(times[-1]) + float(time_step)))
        #spines.append([])
        # add all spines, times in region to the spines and times.
        map(smoothed_spines.append, spines)
        map(smoothed_times.append, eq_times)
    data_type = 'spine'
    if store_in_db:
        description = 'spine smoothed in time, empty strings denote '\
                      'regions that had breaks in them.'
        store_data_in_db(blob_id=blob_id, data_type=data_type, times=times,
                         data=smoothed_spines,
                         description=description, db_doc=db_doc, **kwargs)
    if store_tmp:
        data ={'time':smoothed_times, 'data':smoothed_spines}
        write_tmp_file(data=data, blob_id=blob_id, data_type=data_type)
    return smoothed_times, smoothed_spines

def distance_between_spines(spines1, spines2):
    """
    returns total euclidian distance between all spines in one list and all spines in another list.
    a spine is a list of tuples denoting the points in a spine [(x1,y1), (x2, y2), ... ].

    :param spines1: first list of spines
    :param spines2: second list of spines
    :return: distance
    """
    dist = 0
    for spine1, spine2 in zip(spines1, spines2):
        for pt1, pt2 in zip(spine1, spine2):
            dist += euclidean(pt1, pt2)
    return dist

if __name__ == "__main__":
    #blob_id ='20120914_172813_01708'
    #blob_id = '20121119_162934_07337'
    blob_id = '00000000_000001_00001'
    blob_id = '00000000_000001_00008'
    #blob_id = '20130319_150235_01070'
    #blob_id = '20130320_164252_05955'
    #blob_id = '20130320_153235_40328'
    # HEAD ERRORS
    #blob_id = '20130324_115435_04452'
    #blob_id = '20130320_153237_12115'
    #blob_id = '20130318_153749_02631'
    #blob_id = '20130320_153237_12911'
    #smooth_unflagged_timepoints(blob_id)
    smooth_good_regions_repeatedly(blob_id, repeated_smoothings=1, insert=False)
