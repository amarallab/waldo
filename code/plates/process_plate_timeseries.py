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
from plate_utilities import get_ex_id_files,  write_dset_summary, parse_plate_timeseries_txt_file
from plate_utilities import return_flattened_plate_timeseries, organize_plate_metadata

''' 
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
'''

def process_fitting_for_data_type(dataset, data_type, verbose=True):
    ex_ids, dfiles = get_ex_id_files(dataset, data_type)
    params, errs = [], []
    hours, days = [], []    
    mean_Ns = []
    labels, sublabels, plate_ids = [], [], []

    for i, (ex_id, dfile) in enumerate(izip(ex_ids, dfiles)):
        # get metadata in order
        hour, label, sub_label, pID, day = organize_plate_metadata(ex_id)
        hours.append(hour)
        labels.append(label)
        sublabels.append(sub_label)
        plate_ids.append(pID)
        days.append(day)
        
        times, data = parse_plate_timeseries_txt_file(dfile)
        x, bins, n = rebin_data(times, bins=data, t_step=10)
        y = [np.mean(b) for b in bins]
        mean_Ns.append(np.mean(n))

        # default
        p = [0.0, 0.0, 0.0]
        err = -1.0
        if len(times) > 30000:            
            p, label, err = fit_constrained_decay_in_range(x, y)
            params.append(p)
            errs.append(err)

    data={'ex_ids':ex_ids,
          'hours':hours,
          'fit_params': params,
          'fit_err': err,
          'mean_N':mean_Ns,
          'labels':labels,
          'sub':sublabels,
          'plate_ids':plate_ids,
          'days':days,
          }
    return data


def process_basic_plate_timeseries(dataset, data_type, verbose=True):
    ex_ids, dfiles = get_ex_id_files(dataset, data_type)
    means, stds = [], []
    quartiles = []
    hours = []
    days = []
    #labels, sublabels, plate_ids = {}, {}, {}
    labels, sublabels, plate_ids = [], [], []

    for i, (ex_id, dfile) in enumerate(izip(ex_ids, dfiles)):
        hour, label, sub_label, pID, day = organize_plate_metadata(ex_id)
        hours.append(hour)
        labels.append(label)
        sublabels.append(sub_label)
        plate_ids.append(pID)
        days.append(day)

        flat_data = return_flattened_plate_timeseries(dfile)
        if not len(flat_data):
            continue
        means.append(np.mean(flat_data))
        stds.append(np.std(flat_data))
        men = float(np.mean(flat_data))
        #print men, type(men), men<0
        #print flat_data[:5]
        quartiles.append([stats.scoreatpercentile(flat_data, 25),
                          stats.scoreatpercentile(flat_data, 50),
                          stats.scoreatpercentile(flat_data, 75)])
        if verbose:
            print '{i} {eID} | N: {N} | hour: {h} | label: {l}'.format(i=i, eID=ex_id, N=len(flat_data),
                                                                       h=round(hour, ndigits=1), l=label)                                      

    #for i in zip(ex_ids, means, stds, quartiles):
    #    print i
    data={'ex_ids':ex_ids,
          'hours':hours,
          'mean':means,
          'std':stds,
          'quartiles':quartiles,
          'labels':labels,
          'sub':sublabels,
          'plate_ids':plate_ids,
          'days':days,
          }
    return data

def process_basics_for_standard_set(dataset):
    standard_set = ['cent_speed_bl', 'length_mm', 'curve_bl']
    for data_type in standard_set:
        print dataset, data_type,
        data = process_basic_plate_timeseries(dataset, data_type)
        write_dset_summary(data=data, sum_type='basic', 
                           data_type=data_type, dataset=dataset)


def process_fitting_for_speed(dataset):
    data_type = 'cent_speed_bl'
    data = process_fitting_for_data_type(dataset, data_type)
    write_dset_summary(data=data, sum_type='fit_exponential_decay', 
                       data_type=data_type, dataset=dataset)


if __name__ == '__main__':
    dataset = 'disease_models'
    #data_type = 'cent_speed_bl'
    #data_type = 'length_mm'
    #data_type = 'curve_bl'
    #process_basics_for_standard_set(dataset)
    process_fitting_for_speed(dataset)
