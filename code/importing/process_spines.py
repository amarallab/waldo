#!/usr/bin/env python

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
#from database.mongo_retrieve import mongo_query, unique_blob_ids_for_query
from settings.local import LOGISTICS, FILTER
from import_rawdata_into_db import create_entries_from_blobs_files
from wio.file_manager import get_good_blobs
from wormmetrics.measurement_suite import measure_all, write_plate_timeseries_set, write_plate_percentiles
from centroid import process_centroid

def basic_data_to_smoothspine(blob_id, verbose=True, **kwargs):
    """
    Calls every script needed to take a blob_id from raw-data to the finished smoothed spine.

    :param blob_id: identification string for the blob.
    :param verbose: on/off toggle for messages related to processing.
    """
    kwargs['store_in_db'] = False
    kwargs['store_tmp'] = True
    if verbose:
        print 'Creating Rough Spine from Outline'
    times, treated_spines, bad_times = create_spine_from_outline(blob_id, verbose=verbose, **kwargs)
    if verbose:
        print '\tspine created with {N} time-points.'.format(N=len(treated_spines))
    # calculate necessary measurments to flag wrong shapes
        print 'Computing Rough Measurements'
    compute_basic_measurements(blob_id, **kwargs)
    # flag parts of spine creation process
    flag_blob_id(blob_id, **kwargs)
    # (flags + treated_spine) to (flagged_spine)
    create_breaks_for_blob_id(blob_id, **kwargs)
    if verbose:
        print 'Finalzing Spine for Good Regions'
    smoothed_times, smoothed_spines = smooth_good_regions_repeatedly(blob_id, **kwargs)
    if verbose:
        print '\tfinished smoothing spine | N: {N}'.format(N=len(smoothed_times))
    return smoothed_times, smoothed_spines

def just_process_centroid(ex_id, **kwargs):
    overwrite = kwargs.pop('overwrite', True)
    print kwargs    
    if not overwrite:
        good_blobs = get_good_blobs(ex_id, key='xy')
        if len(good_blobs) > 0:
            print '{eID} already processed. to reprocess use overwrite=True'
            return False
        
    process_ex_id(ex_id, just_centroid=True, **kwargs)            

def process_ex_id(ex_id, debug=False, just_centroid=False, **kwargs):
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


    if not overwrite and len(get_good_blobs(ex_id)) > 0:
        # check if entries for ex_id already exist, dont do anything
        print ex_id, 'already processed'
        return False

    blob_ids = create_entries_from_blobs_files(ex_id, min_body_lengths,
                                               min_duration,
                                               min_size, store_tmp= True,
                                               **kwargs)

    # processing blobs section.
    # must perform: process_spines.process_ex_id
    N = len(blob_ids)
    if N ==0:
        return None
    good_blobs = []
    for i, blob_id in enumerate(sorted(blob_ids)[:], start=1):
        print '################### {id} ({i} of {N}) ###################'.format(i=i, N=N, id=blob_id)
        process_centroid(blob_id, **kwargs)
        if just_centroid:
            continue
        
        times, spines = basic_data_to_smoothspine(blob_id, verbose=True, **kwargs)
        if len(spines) > 0:
            good_blobs.append(blob_id)
        measure_all(blob_id, **kwargs)
        try:
            #basic_data_to_smoothspine(blob_id, verbose=True, **kwargs)
            pass
        except Exception, e:
            e.printStackTrace()
        if debug:
            break
    else:
        if just_centroid:
            return
        write_plate_timeseries_set(ex_id, blob_ids=good_blobs, **kwargs)
        write_plate_percentiles(ex_id, blob_ids=good_blobs, **kwargs)

def all_unprocessed_blob_ids(ex_id, **kwargs):
    # mongo version
    #all_ids = unique_blob_ids_for_query({'data_type': 'metadata'}, **kwargs)
    #processed_ids = unique_blob_ids_for_query({'data_type': 'raw_spine'}, **kwargs)
    #unprocessed_ids = [blob_id for blob_id in all_ids not in processed_ids]
    # flat-file version
    all_blobs = get_good_blobs(ex_id, key='xy_raw')
    processed_blobs = get_good_blobs(ex_id, key='spine')
    unprocessed_ids = list(set(all_blobs) - set(processed_blobs))
    return unprocessed_ids

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # test worm
        blob_id = '00000000_000001_00001'
        blob_id = '00000000_000001_00008'
        blob_id = '20131211_145827_00010'
        # the following three are joined blobs
        blob_id = '20131211_145827_00334'
        blob_id = '20131211_145827_00060'
        #blob_id = '20131211_145827_00212'
        blob_id = '20131211_145827_00010'
        basic_data_to_smoothspine(blob_id, verbose=True)
        #for blob_id in all_unprocessed_blob_ids():
        #    basic_data_to_smoothspine(blob_id)
    else:
        for ex_id in sys.argv[1:]: 
            process_ex_id(ex_id)
            #write_plate_timeseries_set(ex_id, blob_ids=['20131213_140440_06928'])
