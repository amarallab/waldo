#!/usr/bin/env python

'''
Filename: subsample_trait.py
Description: 
Pull one type of data out of database, and save it in jsons organized by ex_id.
data pulled is broken into 15 minute segments. within each 15min segment data is pulled either
by subsampling the data or by binning into 10 second bins.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import datetime
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
import random
import json

# Path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_directory)

# nonstandard imports
from Shared.Code.Database.mongo_retrieve import mongo_query, pull_data_type_for_blob, timedict_to_list
from Import.Code.experiment_index import Experiment_Attribute_Index
from Shared.Code.WormMetrics.switchboard import pull_metric_for_blob_id
from Shared.Code.Settings.data_settings import mongo_settings
import Shared.Code.Database.mongo_support_functions as mongo

# Globals
Data_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Data/Sub-Sample/'

def test_data1():
    """ generate a synthetic dataset. """
    time_dict = {}
    values = [i*0.1 for i in range(3600*10)]
    for v in values:
        tk = str(v).replace('.', '?')
        time_dict[tk] = v
    return time_dict


def bin_blob(blob_id, big_binsize=900, small_binsize=10, final_time=3600, **kwargs):
    """ returns a list mean small bin values for each big bin.
    
    Arguments:
    - `blob_id`:
    - `big_binsize`:
    - `small_binsize`:
    """
    # use the metric switchboard
    speed_dict = pull_metric_for_blob_id(blob_id=blob_id, metric='centroid_speed', remove_skips=True, **kwargs)
    times, speeds = timedict_to_list(speed_dict)
    
    num_big_bins = final_time / big_binsize
    data = [{} for _ in range(num_big_bins)]

    for t,v in izip(times, speeds):
        big_bin = int(t / big_binsize)
        if big_bin >= num_big_bins:
            print 'warning binning is weird for time:', t
            big_bin = num_big_bins -1

        small_bin = int((t - big_bin * big_binsize) /small_binsize)
        if small_bin not in data[big_bin]:
            data[big_bin][small_bin] = []
        data[big_bin][small_bin].append(v)

    means = [[] for _ in range(num_big_bins)]
    for big_bin, bigbin_dataset in enumerate(data):
        for small_bin in sorted(bigbin_dataset):
            means[big_bin].append(np.mean(bigbin_dataset[small_bin]))
    return means

def sample_blob(blob_id, big_binsize=900, sample_prob=0.01, final_time=3600, **kwargs):
    speed_dict = pull_metric_for_blob_id(blob_id=blob_id, metric='centroid_speed', remove_skips=True, **kwargs)
    times, speeds = timedict_to_list(speed_dict)
    num_big_bins = final_time / big_binsize
    data = [[] for _ in range(num_big_bins)]
    for t,v in izip(times, speeds):
        big_bin = int(t / big_binsize)
        if big_bin >= num_big_bins:
            print 'warning binning is weird for time:', t
            big_bin = num_big_bins -1
        data[big_bin].append(v)
    samples_per_bin = [int(len(d) * sample_prob) for d in data]
    samples = [[] for _ in range(num_big_bins)]
    for big_bin, bigbin_dataset in enumerate(data):
        bin_samples = random.sample(bigbin_dataset, samples_per_bin[big_bin])
        samples[big_bin] = bin_samples
    return samples
    
def show_it(datasets, numbins=50):
    fig = plt.figure()
    for dset in datasets:
        if len(dset) > numbins:
            h, bins = np.histogram(dset, bins=numbins)#, range=[0, 3600])
            bincenters = [(bins[i] + bins[i+1])/2 for i in range(numbins)]
            plt.plot(bincenters, h)
    plt.show()

def sample_ex_id(ex_id, big_binsize=900, sample_function=sample_blob, **kwargs):

    # things to pull from the database:
    stage1_data = mongo_query({'ex_id': ex_id, 'type': 'stage1'}, find_one=True, col='plate_collection', **kwargs)
    blob_ids = stage1_data['blob_ids']
    final_time = int(stage1_data['vid-duration'])
    # initialize data structure
    num_big_bins = final_time / big_binsize
    ex_id_data = [[] for _ in range(num_big_bins)]
    
    for blob_id in blob_ids:
        blob_data = sample_function(blob_id, final_time=final_time, **kwargs)
        #show_it(datasets=blob_data)
        for i in range(num_big_bins):
            ex_id_data[i] += blob_data[i]
    return ex_id_data

def choose_ex_id(key='purpose', value='N2_aging'):
    ei = Experiment_Attribute_Index()
    return ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value)

def write_ex_id_subsamples(**kwargs):
    ex_ids = choose_ex_id()
    for ex_id in ex_ids[:]:
        try:
            savename = '{path}{ex_id}_{data_type}_sampleprob0.01_allbins.json'.format(path=Data_DIR, ex_id=ex_id,
                                                                                data_type='centroid_speed')
            print savename
            datasets = sample_ex_id(ex_id=ex_id, sample_function=sample_blob, **kwargs)
            json.dump(datasets, open(savename, 'w'))
        except Exception as e:
            print e
        try:
            savename = '{path}{ex_id}_{data_type}_binsize10_allbins.json'.format(path=Data_DIR, ex_id=ex_id,
                                                                                 data_type='centroid_speed')
            print savename
            datasets = sample_ex_id(ex_id=ex_id, sample_function=bin_blob, **kwargs)
            json.dump(datasets, open(savename, 'w'))
        except Exception as e:
            print e

def write_full_ex_id_data(**kwargs):
    measurement_type = 'centroid_speed'
    ex_ids = choose_ex_id()
    ex_id = ex_ids[0]
    savename = '{path}/{type}/{ex_id}.txt'.format(path=Data_DIR, ex_id=ex_id, type='centroid_speed')
    print savename

def main(**kwargs):
    blob_id = '20130320_102312_10828'
    #blob_id = '20130320_102312_19189'
    ex_id = '20130320_102312'
    #sample_ex_ids_for_query()
    #datasets = bin_blob(blob_id=blob_id)
    #datasets = sample_blob(blob_id=blob_id)
    datasets = sample_ex_id(ex_id=ex_id, sample_function=bin_blob, **kwargs)
    datasets = sample_ex_id(ex_id=ex_id, sample_function=sample_blob, **kwargs)
    show_it(datasets)

if __name__ == '__main__':

    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['blob_collection'])
    try:
        #main(mongo_client=mongo_client)
        #write_ex_id_subsamples(mongo_client=mongo_client)
        write_full_ex_id_data(mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)
