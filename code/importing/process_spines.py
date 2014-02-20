a#!/usr/bin/env python

'''
Filename: process_spines.py
Description: This is a harness that calls every data-processing script needed to make
a smoothed center-line (ie. spine or skeleton) with the head oriented in the correct direction.

For each blob, this involves:
1. creating a basic center-line of 50 points from the outline
2. Using the basic center-line to compute width and length measurements at every time-point
3. Using the width and length measurements to flag time-points that are inconsistent with the other measurements.
4. Using the flags to denote 'good' and 'bad' regions of data for that blob.
5. Smooth the 'good' regions in time and space as well as predict the head of the worm
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import matplotlib.pyplot as plt        

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.abspath(HERE + '/../../')
sys.path.append(PROJECT_DIR)

# nonstandard imports
from create_spine import create_spine_from_outline

from flags_and_breaks import flag_blob_id, create_breaks_for_blob_id
from smooth_spines_in_time import smooth_good_regions_repeatedly
from compute_basic_measurements import compute_basic_measurements
from database.mongo_retrieve import mongo_query, unique_blob_ids_for_query
from settings.local import LOGISTICS, FILTER
from import_rawdata_into_db import create_entries_from_blobs_files
import shared.wio
from measurement_suite import measure_all

def basic_data_to_smoothspine(blob_id, verbose=True, **kwargs):
    """
    Calls every script needed to take a blob_id from raw-data to the finished smoothed spine.

    :param blob_id: identification string for the blob.
    :param verbose: on/off toggle for messages related to processing.
    """
    kwargs['store_in_db'] = False
    kwargs['store_tmp'] = True
    times, treated_spines, bad_times = create_spine_from_outline(blob_id, verbose=verbose, **kwargs)
    if verbose:
        print 'spine created with {N} time-points.'.format(N=len(treated_spines))
    # calculate necessary measurments to flag wrong shapes
    compute_basic_measurements(blob_id, **kwargs)
    if verbose:
        print 'basic measurements calculated'
    # flag parts of spine creation process
    flag_blob_id(blob_id, **kwargs)
    if verbose:
        print 'flag breaks inserted'
    # (flags + treated_spine) to (flagged_spine)
    create_breaks_for_blob_id(blob_id, **kwargs)
    # (flagged_spine) to (smoothed_spine)
    #smooth_unflagged_timepoints(blob_id) # depreciated
    if verbose:
        print 'flags saved'
    smoothed_times, smoothed_spines = smooth_good_regions_repeatedly(blob_id, **kwargs)
    if verbose:
        print 'finished smoothing spine with {N} remaining time-points'.format(N=len(smoothed_times))
        
    return smoothed_times, smoothed_spines

def process_ex_id(ex_id, **kwargs):
    '''
    processes all blobs in database from ex_id from raw data all the way to smoothspines.
    if ex_id does not have any blobs already in database, it reads the .blobs files and inserts
    them before processing them to smoothspines.

    :param ex_id: the identification string for a particular recording.
    '''

    min_body_lengths = kwargs.pop('min_body_lengths', FILTER['min_body_lengths'])
                                  
    min_duration = kwargs.pop('min_duration', FILTER['min_duration'])
    min_size = kwargs.pop('min_size', FILTER['min_size'])
    overwrite = kwargs.pop('overwrite', True)

    # importing blobs section
    # must perform: import_rawdata_into_db.import_ex_id

    # check if entries for ex_id already exist, dont do anything
    if not overwrite:
        entries = mongo_query({'ex_id': ex_id, 'data_type': 'metadata'}, {'blob_id': 1, 'duration': 1}, **kwargs)
        if len(entries) > 0:
            return False
    
    blob_ids = create_entries_from_blobs_files(ex_id, min_body_lengths, min_duration, min_size, store_tmp= True, **kwargs)

    # processing blobs section.
    # must perform: process_spines.process_ex_id
    N = len(blob_ids)
    for i, blob_id in enumerate(sorted(blob_ids)[:], start=1):
        print '################### {id} ({i} of {N}) ###################'.format(i=i, N=N, id=blob_id)
        times, spines = basic_data_to_smoothspine(blob_id, verbose=True, **kwargs)
        measure_all(blob_id, **kwargs)
        try:
            #basic_data_to_smoothspine(blob_id, verbose=True, **kwargs)
            pass
        except Exception as e:
            print e
        print 'den of data', len(times), len(spines)
        good_stuff = [(t, s) for (t, s) in zip(times, spines) if s]
        if len(good_stuff) <= 3:
            continue
        times, spines = zip(*good_stuff)
        x, y = zip(*zip(*spines)[0])
        '''
        try:
        plt.plot(x, y)
        x, y = zip(*zip(*spines)[25])
        plt.plot(x, y)
        x, y = zip(*zip(*spines)[-1])
        plt.plot(x, y)
        plt.savefig(blob_id + '.png')
        plt.clf()
        '''
def all_unprocessed_blob_ids(**kwargs):
    all_ids = unique_blob_ids_for_query({'data_type': 'metadata'}, **kwargs)
    processed_ids = unique_blob_ids_for_query({'data_type': 'raw_spine'}, **kwargs)
    unprocessed_ids = [blob_id for blob_id in all_ids not in processed_ids]
    return unprocessed_ids

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # test worm
        blob_id = '00000000_000001_00001'
        blob_id = '00000000_000001_00008'
        blob_id = '20130319_150235_01070'
        blob_id = '20130319_150235_00014'
        blob_id = '20130319_150235_00426'
        blob_id = '20130319_150235_01501'
        # large bson error:
        #blob_id = '20130331_160517_02379'
        #blob_id = '20130320_164252_05955'
        #blob_id = '20130320_153235_40328'
        # json serializable error
        #blob_id = '20130413_123246_03046'

        #blob_id = '20130324_115435_04452'
        #blob_id = '20130319_150235_01070'

        basic_data_to_smoothspine(blob_id, verbose=True)
        #for blob_id in all_unprocessed_blob_ids():
        #    basic_data_to_smoothspine(blob_id)
    else:
        for ex_id in sys.argv[1:]: process_ex_id(ex_id)

