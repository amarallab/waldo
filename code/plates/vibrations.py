#!/usr/bin/env python

'''
Filename: vibrations.py
Description: Unfinished code for looking at vibrations on a plate.
'''
__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'
# TODO: contains lots of unused imports

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

# TODO: get vibration calculations from blobs files! they contain much higher N. it doesn't matter if the objects move or not.

def testset():
    import random

    numpoints = 100
    numframes = 5
    frames = [random.randrange(0, numframes) for i in range(numpoints)]
    nums = [random.randrange(0, 200) for i in range(numpoints)]

    itermeans = [0 for i in range(numframes)]
    iterN = [0 for i in range(numframes)]
    store_all = [[] for i in range(numframes)]
    for (f, n) in zip(frames, nums):
        itermeans[f], iterN[f] = iterMean(n, itermeans[f], iterN[f])
        store_all[f].append(n)
    for f in range(numframes):
        print itermeans[f], np.mean(store_all[f])
        
def iterMean(item, mean=0, N=0):
    mean +=(item - mean) * (1 - (N / (N + 1.0)))
    return mean, N + 1

def testmeans():    
    import random
    num = 10
    x = [random.randrange(0, 1000) for i in range(num)]
    mean, N = 0, 0
    for i in range(num):
        m1 = np.mean(x[:i+1])
        mean, N = iterMean(x[i], mean, N)
    print mean, np.mean(x)
 
def read_frames_from_summary():
    frames = range(1000)
    times = frames.copy()
    return frames, times
                            
if __name__ == '__main__':
    testset()
    #testmeans()    
'''        
def calculate_vibrations(blob_ids, **kwargs):
    """
    This may be a bad place to do this since there typically arent many blobs being tracked at any give time.

    :param blob_ids:
    :param kwargs:
    :return:
    """
    N_threshold_for_timepoint = 3

    # pool vibrations at every timestep
    dx_pooled_timedict, dy_pooled_timedict, N_dict = {}, {}, {}
    for blob_id in blob_ids:
        print blob_id
        times, blob_dxs, blob_dys = centroid.compute_centroid_measures(blob_id, metric='vibration', **kwargs)
        for (t, dx, dy) in izip(times, blob_dxs, blob_dys):
            if t not in dx_pooled_timedict:
                dx_pooled_timedict[t] = []
            if t not in dy_pooled_timedict:
                dy_pooled_timedict[t] = []
            if t not in N_dict:
                N_dict[t] = 0
            N_dict[t] += 1
            dy_pooled_timedict[t].append(dy)
            dx_pooled_timedict[t].append(dx)


    # take median at each timestep and pool across all timesteps (if N for that timepoint is > theshold)
    dxs, dys = [], []
    for t in times:
        if N_dict[t] >= N_threshold_for_timepoint:
            dxs.append(np.median(dx_pooled_timedict[t]))
            dys.append(np.median(dy_pooled_timedict[t]))

    x_mean, x_std = np.mean(dxs), np.std(dxs)
    y_mean, y_std = np.mean(dys), np.std(dys)
    print x_mean, x_std
    print y_mean, y_std
    print np.mean(N_dict.values()), 'avg N'

    plt.figure()
    plt.plot(dxs, dys)
    plt.show()
'''
