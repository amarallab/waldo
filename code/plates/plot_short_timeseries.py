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


def plot_plate_median(ax, times, median, q1st, q3rd, ground=True, ylabel=''):

    ax.plot(times, median, color='blue')
    ax.plot(times, q1st, color='blue', alpha=0.3)
    ax.plot(times, q3rd, color='blue', alpha=0.3)
    ax.fill_between(times, q1st, q3rd, color='blue', alpha=0.2)
    ax.set_ylim([0, max(q3rd)])
    #plt.xlim([0, max(times)])    


def plot_plate_timeseries(ex_id, data_type, dataset=None, save=False):


    if dataset == None:
        dataset = get_dset(ex_id)


    format_results_filename(ex_id, result_type='timeseries', tag=None,
                            dset=None, ID_type='dset',
                            date_stamp=None,
                            file_type='png',
                            ensure=False)
    print ex_id
    print dataset
    
    times, data = read_plate_timeseries(ex_id, dataset, data_type, tag='timeseries')
    
    not_nan = lambda x: not np.isnan(x)
    data = [filter(not_nan, d) for d in data]

    times, data = zip(*[(t,d) for (t,d) in zip(times, data) if len(d) > 4])

    time_in_minutes = np.array(times) / 60.0

    N = [len(d) for d in data]
    q1st = [stats.scoreatpercentile(d, 25) for d in data if len(d)>4]
    median = [np.median(d) for d in data]
    q3rd = [stats.scoreatpercentile(d, 75) for d in data]


    '''    
    plot_plate(times=times,
               median=median,
               q1stg1=q1st,
               q3rd=q3rd)
    '''
    fig, ax = plt.subplots()
    plot_plate_median(ax, time_in_minutes, median, q1st, q3rd)
    #ax.plot(times, q1st, times, median, times, q3rd)
    #ax.plot(times, median)
    plt.xlabel('time (m)')
    plt.ylabel(data_type)

    plt.show()
    if save:
        savename = format_results_filename(ID=ex_id, result_type='timeseries',
                                       ID_type='plate', dset=dataset)
        plt.savefig(savename)
    plt.clf()


if __name__ == '__main__':
    eid = '20130610_161943'
    eID = '20130318_165649'
    eID = '20130318_153749'

    data_type = 'cent_speed_bl'
    data_type = 'angle_change'
    data_type = 'length_mm'
    data_type = 'curve_w'
    save = False
    plot_plate_timeseries(ex_id=eid, data_type=data_type, save=save)
