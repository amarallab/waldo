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
from mpltools import style

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(os.path.join(HERE, '..'))
SHARED_DIR = os.path.abspath(os.path.join(CODE_DIR, 'shared'))
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.file_manager import format_results_filename, get_dset
from wio.plate_utilities import read_plate_timeseries, get_plate_files
from wormmetrics.measurement_switchboard import FULL_SET, STANDARD_MEASUREMENTS
from annotation.experiment_index import Experiment_Attribute_Index2

# global settings
style.use('ggplot')


def preprocess_plate_timeseries(times, data):

    if data == None or len(data) == 0:
        print 'no data found'
        return None, None
    #print data[:10]
    not_nan = lambda x: not np.isnan(x)
    data = [filter(not_nan, d) for d in data]
    td = [(t,d) for (t,d) in zip(times, data) if len(d) > 4]
    if td == None or len(td) == 0:
        print 'not enough data collected per timepoint'
        return None, None
    times, data = zip(*td)
    return times, data

def pick_ticks(maxdata, ndig=1, step_inc=0.1):

    if maxdata <= (10**(-1*ndig)):
        ndig += 1
        step_inc = step_inc / 10.0
        return pick_ticks(maxdata, ndig, step_inc)    
    
    maxtick = round(maxdata, ndigits=ndig)
    #print maxtick
    
    #print maxtick
    tick_options = []
    for i in range(1, 5):
        step = i * step_inc
        #print step
        ticks = np.arange(0, maxtick+(10**(-1*ndig)), step=step)
        if round(ticks[-1], ndigits=ndig+1) == round(maxtick, ndigits=ndig+1):
            if 2 <= len(ticks) <=4:
                return ticks


    for i in range(1, 5):
        step = i * step_inc
        ticks = np.arange(0, maxtick+(10**(-1*ndig)), step=step)
        if 2 <= len(ticks) <=4:
            return ticks
    return [0.0, maxtick]


def add_plate_median(ax, ex_id, dataset, data_type):
    print '\tadding', data_type
    times, data = read_plate_timeseries(ex_id, dataset, data_type, tag='timeseries')    
    times, data = preprocess_plate_timeseries(times, data)
    if times == None or data == None:
        ax.set_yticks([])
        return
    time_in_minutes = np.array(times) / 60.0
    times = time_in_minutes

    N = [len(d) for d in data]
    q1st = [stats.scoreatpercentile(d, 25) for d in data if len(d)>4]
    median = [np.median(d) for d in data]
    q3rd = [stats.scoreatpercentile(d, 75) for d in data]

    color = ax._get_lines.color_cycle.next()

    ax.set_ylabel(data_type)
    ax.plot(times, median, color=color)
    #ax.plot(times, q1st, color=color, alpha=0.3)
    #ax.plot(times, q3rd, color=color, alpha=0.3)
    ax.fill_between(times, q1st, q3rd, color=color, alpha=0.2)
    #ax.set_yticks(pick_ticks(max(q3rd)))
    ax.set_xlim([0, max(times)])

    if data_type != 'angle_change':
        ax.set_ylim([0, stats.scoreatpercentile(q3rd, 99)])
    else:
        ax.set_ylim([stats.scoreatpercentile(q1st, 01), stats.scoreatpercentile(q3rd, 99)])        



def plot_all_for_dset(dataset, data_types=STANDARD_MEASUREMENTS, save=False):
    ei = Experiment_Attribute_Index2(dataset)
    ex_ids = list(ei.index)    

    check_dtype = data_types[0]
    print check_dtype
    processed_ex_ids, _ = get_plate_files(dataset, data_type=check_dtype)
    #print ex_ids[:10]
    #print processed_ex_ids[:10]
    print '{N} recordings in {ds} dataset'.format(N=len(ex_ids), ds=dataset)
    print '{N} recordings processed'.format(N=len(processed_ex_ids))

    #worthy_ex_ids, ages = [], []
    for eID, age in zip(ex_ids, ei['age']):
        #plotting2(ex_id=eID, data_types=FULL_SET, save=save)        
        if eID not in processed_ex_ids:
            continue

        print 'checking', eID, 'age:', age
        times, data = read_plate_timeseries(eID, dataset, data_type=check_dtype, tag='timeseries')    
        times, data = preprocess_plate_timeseries(times, data)
        if times == None or data == None:
            continue

        print '\tfound worthy'

        plot_one_plate(eID, data_types, dataset, save, age)
        plt.clf()
        #worthy_ex_ids.append(eID)
        #ages.append(age)

        #print '{N} recordings have sufficient data'.format(N=len(worthy_ex_ids))        
        #for eID, age in zip(worthy_ex_ids, ages):

def plot_one_plate(ex_id, data_types=STANDARD_MEASUREMENTS, dataset=None, save=False, age=''): 
    N = len(data_types)
    if dataset == None:
        dataset = get_dset(ex_id)

    fig, axes = plt.subplots(N, 1, sharex=True)
    for dtype, ax in zip(data_types, axes):
        add_plate_median(ax, ex_id, dataset, dtype)
    plt.xlabel('time (m)')
    #plotting2(ex_id=eID, data_types=STANDARD_MEASUREMENTS)        
    axes[0].set_title(dataset + ' ' + ex_id + ' ' + age)
    #plt.tight_layout()
    if save:
        savename = format_results_filename(ID=ex_id, result_type='timeseries',
                                           ID_type='plate', dset=dataset,
                                           ensure=True)
        print 'saving:'
        print savename
        plt.savefig(savename)
        plt.clf()

if __name__ == '__main__':
    dset = 'N2_aging'
    #plot_all_for_dset(dataset=dset, save=True)

    #save = True
    save = False
    eID = '20130414_140704'
    #eID = '20130318_131056'
    plot_one_plate(ex_id=eID, data_types=FULL_SET, save=save)
    plt.show()

    #plot_all_for_dset('N2_aging', save=save)
