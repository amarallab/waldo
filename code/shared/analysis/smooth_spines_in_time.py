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

#from scipy.spatial.distance import euclidean

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIRECTORY = os.path.abspath(HERE + '/../../')
SHARED_DIRECTORY = PROJECT_DIRECTORY + '/code/shared/'
assert os.path.exists(PROJECT_DIRECTORY), 'project directory not found'
assert os.path.exists(SHARED_DIRECTORY), 'shared code directory not found'
sys.path.append(SHARED_DIRECTORY)

# nonstandard imports
from conf import settings
from wio.file_manager import get_timeseries, write_timeseries_file, get_metadata
from flags_and_breaks import good_segments_from_data, get_flagged_times
import equally_space as es

# set defaults from settings file
TIME_ORDER = settings.SMOOTHING['time_order']
TIME_WINDOW = settings.SMOOTHING['time_window']
SPINE_ORDER = settings.SMOOTHING['spine_order']
SPINE_WINDOW = settings.SMOOTHING['spine_window']
T_STEP = settings.SMOOTHING['time_step']
N_POINTS = settings.SMOOTHING['N_points']


def smooth_good_regions_repeatedly(blob_id, repeated_smoothings=5,
                                   spine_order=SPINE_ORDER,
                                   spine_window=SPINE_WINDOW,
                                   time_order=TIME_ORDER,
                                   time_window=TIME_WINDOW,
                                   store_tmp=True,
                                   time_step=T_STEP, **kwargs):
    """
    Returns a time-dict of spines that have been (1.) repeatedly smoothed in space and then in time
    (2.) oriented such that the majority of movement goes towards the front of the spine.

    :param blob_id: blob_id used to
    :param repeated_smoothings: number of times smoothing process should be repeated. (int)
    :param spine_order:
    :param spine_window:
    :param time_order:
    :param time_window:
    :param insert: toggle if resulting data should be inserted into database.
    :return: timedict of smooothed spines.
    """

    # get data into proper form
    break_list = get_metadata(blob_id, data_type='breaks')
    times, spines = get_timeseries(blob_id, data_type='spine_rough')
    flagged_times = get_flagged_times(blob_id)
    good_regions = good_segments_from_data(break_list, times=times, data=spines,
                                           flagged_times=flagged_times)

    # initialize buckets
    smoothed_times, smoothed_spines = [], []
    # each region is smoothed independently
    for i, region in enumerate(good_regions, start=1):
        # if region is too small, it is not worth smoothing
        safety_factor = 1.3
        if len(region) < time_window * safety_factor:
            continue
        times, spines = zip(*region)
        s, e, N = times[0], times[-1], len(region)
        print '\tregion {i}'.format(i=i)
        print '\tstart: {s} | end: {e} | N: {n}'.format(s=s, e=e, n=N)
        # transform spine point format into matrix format
        times, x_matrix, y_matrix = es.create_spine_matricies(times, spines)
        # smooth once in both directions and make sure points are equally spaced along spine
        x_matrix, y_matrix = es.smooth_matricies_cols(x_matrix, y_matrix, window=time_window, order=time_order)
        x_matrix, y_matrix = es.smooth_matricies_rows(x_matrix, y_matrix, window=spine_window, order=spine_order)
        x_matrix, y_matrix = es.equally_space_matrix_distances(x_matrix, y_matrix)
        # interpolate missing times
        eq_times = es.equally_spaced_tenth_second_times(start=times[0], end=times[-1])
        x_matrix, y_matrix = es.equally_space_matricies_times(eq_times, times, x_matrix, y_matrix)
        # now that times have been set, smooth + space spines repeatedly

        for i in range(repeated_smoothings):
            x_matrix, y_matrix = es.smooth_matricies_cols(x_matrix, y_matrix, window=time_window, order=time_order)
            x_matrix, y_matrix = es.smooth_matricies_rows(x_matrix, y_matrix, window=spine_window, order=spine_order)
            x_matrix, y_matrix = es.equally_space_matrix_distances(x_matrix, y_matrix)
        # check if head is correct and reverse row orientation if not
        x_matrix, y_matrix = es.set_matrix_orientation(x_matrix, y_matrix)
        # transform spine matrix format into point format
        spines = es.spine_matricies_to_points(x_matrix, y_matrix)
        # add an empty string one space after end of region.
        #times.append(str(float(times[-1]) + float(time_step)))
        #spines.append([])
        # add all spines, times in region to the spines and times.
        map(smoothed_spines.append, spines)
        map(smoothed_times.append, eq_times)
    data_type = 'spine'
    if store_tmp:
        write_timeseries_file(ID=blob_id, data_type=data_type,
                              times=smoothed_times, data=smoothed_spines)
    return smoothed_times, smoothed_spines

if __name__ == "__main__":
    #blob_id ='20120914_172813_01708'
    #blob_id = '20121119_162934_07337'
    blob_id = '00000000_000001_00001'
    blob_id = '00000000_000001_00008'
    blob_id = '20130319_150235_01830'
    blob_id = '20130319_150235_00002'
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
