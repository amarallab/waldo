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
from wio.file_manager import ensure_dir_exists
from plate_utilities import get_ex_id_files, TIME_SERIES_DIR, PLATE_DIR, write_plate_summary

FITS_DIR = PLATE_DIR + '/Fits/'
print PLATE_DIR
print FITS_DIR
print TIME_SERIES_DIR

def parse_plate_timeseries_txt_file(dfile):
    times, data = [], []
    with open(dfile) as f:
        for line in f:
            line = line.strip().split(',')
            times.append(float(line[0]))
            data.append(map(float, line[1:]))
    return times, data

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
    ''' no function is fit to data, shows N/bin and values over time. '''    
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

if __name__ == '__main__':
    dataset = 'disease_models'
    data_type = 'cent_speed_bl'

    fit_tag = 'exponential_decay' #+ '-constrained'
    calculate_fit = False
    calculate_basics = True
    
    for i, (times, data, ex_id) in enumerate(read_plate_timeseries(dataset, data_type)):        
        if calculate_basics:
            basic_summary = process_basic_plate_timeseries(times, data, ex_id)
            write_plate_summary(data=basic_summary, ex_id=ex_id, sum_type='basic', 
                                data_type=data_type, dataset=dataset)
        if calculate_fit:
            fitting_summary = process_plate_timeseries(times, data, ex_id)
            write_plate_summary(data=fitting_summary, ex_id=ex_id, sum_type=fit_tag, 
                                data_type=data_type, dataset=dataset)

