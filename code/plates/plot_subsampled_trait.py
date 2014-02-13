#!/usr/bin/env python

'''
Filename: plot_subsample_trait.py
Description: reads json files and creates distributions of speeds for each age.
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
import glob

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
PLOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Results/Sub-Sample/'

def order_ex_ids_by_age(ex_ids):
    """ returns a dictionary in which age is the key and a list of ex_ids is the value for each key"""
    ei = Experiment_Attribute_Index()
    ages = ei.return_attribute_for_ex_ids(ex_ids, attribute='age')
    ex_ids_by_age = {}
    for e, a in izip(ex_ids, ages):
        if a not in ex_ids_by_age:
            ex_ids_by_age[a] = []
        ex_ids_by_age[a].append(e)
    return ex_ids_by_age

def order_files_by_ex_ids(binned=True):
    """ returns a dictionary in which ex_id is the key and the path to the cooresponding json file is the value """
    if binned:
        files = glob.glob('{path}*binsize10*.json'.format(path=Data_DIR))
    else:
        files = glob.glob('{path}*prob*.json'.format(path=Data_DIR))
    files_by_ex_id = {}
    for f in files:
        ex_id = f.split('/')[-1].split('_cent')[0]
        files_by_ex_id[ex_id] = f
    return files_by_ex_id

def show_it(datasets, title, labels, savename=''):
    """ simple function to plot distributions """
    fig = plt.figure()
    plt.title(title)
    for (l, dset) in izip(labels, datasets):
        numbins = 50
        if len(dset) > 50:
            h, bins = np.histogram(dset, bins=numbins, density=True)
            bincenters = [(bins[i] + bins[i+1])/2 for i in range(numbins)]
            plt.plot(bincenters, h, label=l)
            
    plt.legend()
    plt.xlabel('centroid speed (mm/s)')
    plt.xlim([0, 0.3])
    plt.savefig(savename)
    plt.clf()

def main():
    """
    """
    # toggle to go back and forth between random sampling and 
    binned = False

    ex_id_to_file = order_files_by_ex_ids(binned=binned)
    age_to_ex_id = order_ex_ids_by_age(ex_ids=ex_id_to_file.keys())

    # setup graph titles and savenames
    if binned:
        sampling = 'bin size = 10 seconds'
        save_tag = '10sec_bins'
    else:
        sampling = '1/100th of points sampled'
        save_tag = 'subsample100th'

    for age in ['A{a}'.format(a=a) for a in range(1,13)]:
        print age
        # setup four empty bins, then add the bins from ex_id jsons.
        datasets = [[] for _ in range(4)]
        for ex_id in age_to_ex_id[age]:
            f = ex_id_to_file[ex_id]
            ex_data = json.load(open(f, 'r'))
            for i, data in enumerate(ex_data):
                #print i, type(data), len(data)
                datasets[i] += ex_data[i]

        title = 'N2 centriod speed, {sampling}, age {a}'.format(sampling=sampling, a=age)
        savename = '{path}N2_centroid_speed_{t}_age={a}.png'.format(path=PLOT_DIR, a=age, t=save_tag)
        show_it(datasets, title=title, labels=['1', '2', '3', '4'], savename=savename)
 


if __name__ == '__main__':
    main()
