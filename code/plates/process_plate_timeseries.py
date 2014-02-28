#!/usr/bin/env python
'''
Filename: fit_plate_timeseries.py
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
import glob
import scipy.stats as stats

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from exponential_fitting import fit_exponential_decay_robustly, rebin_data, exponential_decay, fit_constrained_decay_in_range
from wio.file_manager import ensure_dir_exists#, get_ex_id_metadata
from plate_utilities import get_ex_id_files, TIME_SERIES_DIR, PLATE_DIR, write_dset_summary
from annotation.experiment_index import Experiment_Attribute_Index

FITS_DIR = PLATE_DIR + '/Fits/'
print PLATE_DIR
print FITS_DIR
print TIME_SERIES_DIR

'''
def read_plate_timeseries(dataset, data_type, is_json=None):
    ex_ids, dfiles = get_ex_id_files(dataset, data_type)
    for ex_id, dfile in izip(ex_ids, dfiles):            
        times, data = parse_plate_timeseries_txt_file(dfile)
        yield times, data, ex_id
 
def fit_function_plot(x, y, xfit, yfit, residuals, n, ex_id, 
                      ylabel='centroid speed (mm/s)', show=False, plot_dir=None):
    # plot 1 shows N/bin over time
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(x, n)
    plt.ylabel('N / bin')   
    plt.title(ex_id)
    # plot 2 shows xy + fit
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(x, y)
    plt.ylabel(ylabel)
    plt.plot(xfit, yfit)
    # plot 3 shows fit residuals
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)            
    plt.plot(x, residuals)
    plt.plot([x[0], x[-1]], [0, 0], lw=2, color='red')
    plt.ylabel('fit residuals')
    # finish formating plot, save and/or show
    plt.xlabel('time (s)')
    if show:
        plt.show()
    if plot_dir:
        ensure_dir_exists(plot_dir)
        fig_name = plot_dir + ex_id + '.png'
        plt.savefig(fig_name)
    plt.clf()

def basic_plot(x, y, n, ex_id, ylabel='', show=False, plot_dir=None):
    # no function is fit to data, shows N/bin and values over time.
    num_plots = 2        
    ax1 = plt.subplot(num_plots, 1, 1)
    plt.plot(x, n)
    plt.ylabel('N / bin')   
    plt.title(ex_id)
    ax2 = plt.subplot(num_plots, 1, 2, sharex=ax1)
    plt.plot(x, y)
    plt.ylabel(ylabel)
    plt.xlabel('time (s)')
    if show:
        plt.show()
    if plot_dir:
        ensure_dir_exists(plot_dir)
        fig_name = plot_dir + ex_id + '.png'
        plt.savefig(fig_name)
    plt.clf()

def process_plate_timeseries(times, data, ex_id, plot_dir=None, fit_to_function=False):
    summary = {'params':[], 'guess':'', 'err':None, 'mean_N':0, 'mean':None, 'std':None, 'quartiles':[]}
    
    if True:
        if not len(times):
            return None
        print times[0], 'to', times[-1], len(times), 'points'

        
        flat_data = []
        for i, t_bin in enumerate(data):
            flat_data += list(t_bin)            
        flat_data = np.array(flat_data)
        print 'total measurements: {N}'.format(N= len(flat_data))
        summary['mean'] = np.mean(flat_data)
        summary['std'] = np.std(flat_data)
        summary['quartiles'] = [stats.scoreatpercentile(flat_data, 25),
                                stats.scoreatpercentile(flat_data, 50),
                                stats.scoreatpercentile(flat_data, 75)]
                                                       
        x, bins, n = rebin_data(times, bins=data, t_step=10)
        y = [np.mean(b) for b in bins]
        summary['mean_N'] = np.mean(n)

        # if almost all timepoints present, attempt to fit to exponential decay
        if fit_to_function and len(times) > 30000:
            #p, label, err = fit_exponential_decay_robustly(x, y)
            p, label, err = fit_constrained_decay_in_range(x, y)
            summary['params'], summary['guess'], summary['err'] = list(p), label, err
            xfit = np.linspace(min(x), max(x), 3000)
            yfit = exponential_decay(p, xfit)
            residuals = np.array(y) -  np.array(exponential_decay(p, x))
            fit_function_plot(x, y, xfit, yfit, residuals, n, ex_id, ylabel='centroid speed (mm/s)', show=False, plot_dir=plot_dir)
        else:
            basic_plot(x, y, n, ex_id, ylabel='length (mm)', show=False, plot_dir=plot_dir)
    try:
        pass
    except Exception as e:
        print ex_id, 'failed', summary
        print e
        return None
    return summary

def process_basic_plate_timeseries(times, data, ex_id, plot_dir=None, fit_to_function=False):
    summary = {'ex_id':ex_id, 'mean_N':0, 'mean':None, 'std':None, 'quartiles':[]}
    
    if True:
        if not len(times):
            return None
        print times[0], 'to', times[-1], len(times), 'points'        
        flat_data = []
        for i, t_bin in enumerate(data):
            flat_data += list(t_bin)            
        flat_data = np.array(flat_data)
        print 'total measurements: {N}'.format(N= len(flat_data))
        summary['mean'] = np.mean(flat_data)
        summary['std'] = np.std(flat_data)
        summary['quartiles'] = [stats.scoreatpercentile(flat_data, 25),
                                stats.scoreatpercentile(flat_data, 50),
                                stats.scoreatpercentile(flat_data, 75)]                                                       
        x, bins, n = rebin_data(times, bins=data, t_step=10)
        y = [np.mean(b) for b in bins]
        summary['mean_N'] = np.mean(n)
    try:
        pass
    except Exception as e:
        print ex_id, 'failed', summary
        print e
        return None
    return summary
'''

def ex_id_to_datetime(ex_id):
    ''' converts an experiment id to a datetime object '''     
    parts = ex_id.split('_')
    if len(parts) != 2:
        print 'Error: something is off with this ex_id', ex_id
        return None
    yearmonthday, hourminsec = parts
    year, month, day = map(int, [yearmonthday[:4], yearmonthday[4:6], yearmonthday[6:]])
    h, m, s = map(int, [hourminsec[:2], hourminsec[2:-2], hourminsec[-2:]])
    return datetime.datetime(year, month, day, h, m, s)


def parse_plate_timeseries_txt_file(dfile):
    times, data = [], []
    with open(dfile) as f:
        for line in f:
            line = line.strip().split(',')
            times.append(float(line[0]))
            data.append(map(float, line[1:]))
    return times, data

def organize_plate_metadata(ex_id):
    ei = Experiment_Attribute_Index()

    m = ei.return_attributes_for_ex_id(ex_id)
    label = m.get('label', 'label')
    sub_label = m.get('sublabel', 'set')
    sub_label = '{l}-{sl}'.format(l=label, sl=sub_label)
    pID = m.get('plate-id', 'set B')


    recording_time = ex_id
    plating_time = m.get('l1-arrest', None)
    #print 'plated at:', plating_time
    #print 'recorded at:', recording_time
    age = 0
    if recording_time and plating_time:
        t0 = ex_id_to_datetime(plating_time)
        t1 = ex_id_to_datetime(recording_time) 
        age = (t1 - t0).total_seconds()/3600.


    #age = '{et} - {pt}'.format(et=recording_time, pt=plating_time)
    #for i in m:
    #    print i
    return age, label, sub_label, pID

def process_basic_plate_timeseries(dataset, data_type):
    ex_ids, dfiles = get_ex_id_files(dataset, data_type)
    means, stds = [], []
    quartiles = []
    ages = []
    #labels, sublabels, plate_ids = {}, {}, {}
    labels, sublabels, plate_ids = [], [], []

    for i, (ex_id, dfile) in enumerate(izip(ex_ids, dfiles)):
        age, label, sub_label, pID = organize_plate_metadata(ex_id)
        ages.append(age)
        labels.append(label)
        sublabels.append(sub_label)
        plate_ids.append(pID)
        '''
        # divide plates by labels
        if label not in labels:
            labels[label] = []
        labels[label].append(ex_id)

        # alternate division with more subcatagories
        if sub_label not in sublabels:
            sublabels[sub_label] = []
        sublabels[sub_label].append(ex_id)        
        
        # individual plates grouped together by first appearance
        if pID not in plate_ids:
            plate_ids[pID] = []
        plate_ids[pID].append(ex_id)        
        '''
        times, data = parse_plate_timeseries_txt_file(dfile)
        flat_data = []

        if not len(times):
            continue
        print '{i} {eID} | N: {N} | age: {a} | label: {l}'.format(i=i, eID=ex_id, N=len(times),
                                                            a=round(age, ndigits=1), l=label)                                      
        for i, t_bin in enumerate(data):
            flat_data += list(t_bin)
        # take data out of bins and remove nan values
        flat_data = np.array(flat_data)
        N_w_nan = len(flat_data)
        flat_data = flat_data[np.logical_not(np.isnan(flat_data))]
        N_wo_nan = len(flat_data)
        if N_wo_nan != N_wo_nan:
            print '{N} nans removed'.format(N=N_w_nan-N_wo_nan)
        means.append(np.mean(flat_data))
        stds.append(np.std(flat_data))
        men = float(np.mean(flat_data))
        #print men, type(men), men<0
        #print flat_data[:5]
        quartiles.append([stats.scoreatpercentile(flat_data, 25),
                          stats.scoreatpercentile(flat_data, 50),
                          stats.scoreatpercentile(flat_data, 75)])
    #for i in zip(ex_ids, means, stds, quartiles):
    #    print i
    data={'ex_ids':ex_ids,
          'ages':ages,
          'mean':means,
          'std':stds,
          'quartiles':quartiles,
          'labels':labels,
          'sub':sublabels,
          'plate_ids':plate_ids,
          }

    return data

if __name__ == '__main__':
    dataset = 'disease_models'
    data_type = 'cent_speed_bl'
    #data_type = 'length_mm'
    #data_type = 'curve_bl'
    fit_tag = 'exponential_decay' #+ '-constrained'
    calculate_fit = False
    calculate_basics = True


    if calculate_basics:
        print dataset, data_type,
        data = process_basic_plate_timeseries(dataset, data_type)
        write_dset_summary(data=data, sum_type='basic', 
                           data_type=data_type, dataset=dataset)

