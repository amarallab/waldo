#!/usr/bin/env python

'''
Filename: create_plate_document.py
Description: inserts documents into the plate collection in the worm_db database.
'''
__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
from itertools import izip
import numpy as np
import summary_reader as reader
import glob
import json
import scipy.stats as stats

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
print project_directory
sys.path.append(project_directory)

# nonstandard imports

import Shared.Code.Database.mongo_support_functions as mongo
from Shared.Code.Database.mongo_retrieve import mongo_query, pull_data_type_for_blob, timedict_to_list
from Shared.Code.Settings.data_settings import mongo_settings

from Shared.Code.Database.mongo_insert import insert_plate_document
import Shared.Code.WormMetrics.centroid_measures as centroid
import Shared.Code.WormMetrics.spine_measures as spine
from Shared.Code.Settings.data_settings import logistics_settings
from Import.Code.experiment_index import Experiment_Attribute_Index
import matplotlib.pyplot as plt

DATA_DIR = logistics_settings['filesystem_data']


def choose_ex_id(key='purpose', value='N2_aging'):
    ei = Experiment_Attribute_Index()
    return ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value)

# TODO: I have not yet added in a stage2 data type for the plate collection. not quite sure what to add.
def calculate_binned_spinespeeds(blob_ids, binsize=10, **kwargs):
    # remove this when it is funcitonal code
    assert False
    # pool vibrations at every timestep
    pooled_speeds, N_points, N_blobs = {}, {}, {}
    for blob_id in blob_ids:
        print blob_id
        time_dict = spine.compute_spine_measures(blob_id, metric='speed_along', datatype='smoothed_spine', **kwargs)
        times, speeds = timedict_to_list(time_dict)
        for time, speed in izip(times, speeds):
            tbin = int(time / binsize) * binsize
            if tbin not in pooled_speeds:
                pooled_speeds[tbin] = []
            if tbin not in N_points:
                N_points[tbin] = 0
            if tbin not in N_blobs:
                N_blobs[tbin] = []

            pooled_speeds[tbin].append(speed)
            N_points[tbin] += 1
            if blob_id not in N_blobs[tbin]:
                N_blobs[tbin].append(blob_id)
                        
    ts, y, Ns = [], [], []
    for t in sorted(pooled_speeds):
        print t, np.median(pooled_speeds[t])
        ts.append(t)
        y.append(np.median(pooled_speeds[t]))
        Ns.append(len(set(N_blobs[t])))

    plt.figure()
    plt.plot(ts, y)
    plt.plot(ts, Ns)
    plt.show()

    return 


def calculate_binned_centroidspeeds(blob_ids, binsize=1, **kwargs):
    """
    creates a dictionary containing binned data for number of worms tracked at stage1.
    """
    # pool vibrations at every timestep
    pooled_speeds, N_points, N_blobs = {}, {}, {}
    for blob_id in blob_ids:
        print blob_id
        time_dict = centroid.compute_centroid_measures(blob_id, metric='centroid_speed', smooth=True,  **kwargs)
        times, speeds = timedict_to_list(time_dict)
        for time, speed in izip(times, speeds):
            if binsize == 0:
                tbin = time
            else:
                tbin = int(time / binsize) * binsize
            if tbin not in pooled_speeds:
                pooled_speeds[tbin] = []
            if tbin not in N_points:
                N_points[tbin] = 0
            if tbin not in N_blobs:
                N_blobs[tbin] = []
            pooled_speeds[tbin].append(speed)
            N_points[tbin] += 1
            if blob_id not in N_blobs[tbin]:
                N_blobs[tbin].append(blob_id)

    # calculate deciles for each timepoint
    data = {'time':[], 'N':[], 'median':[], '1st_q':[], '3rd_q':[]}
    for t in sorted(pooled_speeds):
        #speed_declies =  [stats.scoreatpercentile(pooled_speeds[t], p) for p in range(10, 100, 10)]
        data['time'].append(t)
        data['1st_q'].append(stats.scoreatpercentile(pooled_speeds[t], 25))
        data['median'].append(stats.scoreatpercentile(pooled_speeds[t], 50))
        data['3rd_q'].append(stats.scoreatpercentile(pooled_speeds[t], 75))
        data['N'].append(len(set(N_blobs[t])))
    return data

def create_plate_doc(ex_id, doc_type, data, blobs=[], ei=None, **kwargs):
    """
    creates and inserts a document into the plate collection in the worm database.

    Arguments:
    - `ex_id`: plate ID
    - `doc_type`: key field to index in database (field name: 'type')
    - `data`: a dictionary that will be inserted into the 'data' field in the document
    - `ei`: an Experiment_Attribute_Index object. if not specified a new one will be created.
    - `**kwargs`: used for database communication
    """
    #  initialize with all metadata
    if not ei:
        ei = Experiment_Attribute_Index()
    new_doc = ei.return_attributes_for_ex_id(ex_id)
    print new_doc
    new_doc['ex_id'] = ex_id
    new_doc['type'] = doc_type
    new_doc['data'] = data
    new_doc['blob_ids'] = blobs
    insert_plate_document(new_doc, **kwargs)

def create_stage0_plate_doc(ex_id, **kwargs):
    """
    finds the .summary file for the plate, collects all standard info and stores in into the plate collection.

    :param ex_id - the ID of the plate
    :param ei - an Experiment_Attribute_Index object
    """
    path = DATA_DIR + ex_id + '/*.summary'
    sum_files = glob.glob(path)
    if len(sum_files) > 1:
        print 'weird, there are a lot of sum files', sum_files
    data = reader.read_summary(sum_files[0])
    create_plate_doc(ex_id=ex_id, doc_type='stage0', data=data, **kwargs)

def create_stage1_plate_doc(ex_id, **kwargs):
    """
    finds the stage1 blobs in the database, calculates N and median speed at each timepoint,
    creates a list of all blob_ids for that plate and creates a document for the plate collection with all this info.

    :param ex_id - the ID of the plate
    """

    blob_ids = [e['blob_id'] for e in mongo_query({'ex_id': ex_id, 'data_type': 'metadata'}, {'blob_id':1}, **kwargs)]
    data= calculate_binned_centroidspeeds(blob_ids, **kwargs)
    create_plate_doc(ex_id=ex_id, doc_type='stage1', data=data, blobs=blob_ids, **kwargs)


def create_stage2_plate_doc(ex_id, ei=None, **kwargs):
    # TODO: nonfunctional code. need to decide what I want in data field.
    assert False
    blob_ids = [e['blob_id'] for e in mongo_query({'ex_id': ex_id, 'data_type': 'smoothed_spine'}, {'blob_id':1}, **kwargs)]
    blob_ids = list(set(blob_ids))
    data=calculate_binned_spinespeeds(blob_ids, **kwargs)
    create_plate_doc(ex_id=ex_id, doc_type='stage2', data=data, blobs=blob_ids, **kwargs)

def main(ex_ids=[]):
    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['blob_collection'])
    try:
        for ex_id in ex_ids:
            #create_plate_doc(ex_id=ex_id, mongo_client=mongo_client)
            create_stage0_plate_doc(ex_id=ex_id, mongo_client=mongo_client)
            create_stage1_plate_doc(ex_id=ex_id, mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)

if __name__ == '__main__':
    ex_ids = choose_ex_id()
    main(ex_ids=ex_ids[:])
    #main(ex_ids=['20130325_152726'])

