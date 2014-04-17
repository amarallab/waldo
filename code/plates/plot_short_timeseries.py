#!/usr/bin/env python

'''
Filename: plot_plate_timeseries.py
Description:  
'''
__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import pandas as pd
from itertools import izip
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(os.path.join(HERE, '..'))
SHARED_DIR = os.path.abspath(os.path.join(CODE_DIR, 'shared'))
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.file_manager import format_results_filename, get_dset
from wio.plate_utilities import read_plate_timeseries
from wormmetrics.measurement_switchboard import FULL_SET, STANDARD_MEASUREMENTS
from annotation.experiment_index import Experiment_Attribute_Index2

def add_plate_median(ax, ex_id, dataset, data_type):
    times, data = read_plate_timeseries(ex_id, dataset, data_type, tag='timeseries')    
    ax.set_ylabel(data_type)
    print ex_id, dataset, data_type
    if data == None or len(data) == 0:
        print 'no data found'
        return
    not_nan = lambda x: not np.isnan(x)
    data = [filter(not_nan, d) for d in data]
    td = [(t,d) for (t,d) in zip(times, data) if len(d) > 4]
    if td == None or len(td) == 0:
        print 'no data retained'
        return
    
    times, data = zip(*td)
    time_in_minutes = np.array(times) / 60.0
    times = time_in_minutes

    N = [len(d) for d in data]
    q1st = [stats.scoreatpercentile(d, 25) for d in data if len(d)>4]
    median = [np.median(d) for d in data]
    q3rd = [stats.scoreatpercentile(d, 75) for d in data]

    ax.plot(times, median, color='blue')
    ax.plot(times, q1st, color='blue', alpha=0.3)
    ax.plot(times, q3rd, color='blue', alpha=0.3)
    ax.fill_between(times, q1st, q3rd, color='blue', alpha=0.2)
    ax.set_ylim([0, max(q3rd)])


def plotting2(ex_id, data_types, dataset=None, save=False):    
    if dataset == None:
        dataset = get_dset(ex_id)

    N = len(data_types)
    fig, axes = plt.subplots(N, 1)

    for dtype, ax in zip(data_types, axes):
        add_plate_median(ax, ex_id, dataset, dtype)
    axes[0].set_title(dataset + ' ' + ex_id)
    plt.xlabel('time (m)')


    

    if save:
        savename = format_results_filename(ID=ex_id, result_type='timeseries',
                                       ID_type='plate', dset=dataset)
        plt.savefig(savename)


def plot_all_for_dset(dataset, save=False):
    ei = Experiment_Attribute_Index2(dataset)
    for eID in ei.index:
        #plotting2(ex_id=eID, data_types=FULL_SET, save=save)        
        plotting2(ex_id=eID, data_types=STANDARD_MEASUREMENTS, save=save)        
        plt.show()
        plt.clf()

if __name__ == '__main__':
    eID = '20130610_161943'
    #eID = '20130318_165649'
    #eID = '20130318_153749'

    save = False
    plotting2(ex_id=eID, data_types=FULL_SET, save=save)
    plt.show()

    #plot_all_for_dset('N2_aging', save=save)
