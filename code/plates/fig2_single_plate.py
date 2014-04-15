#!/usr/bin/env python
'''
Filename: fig2_single_plate.py

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
CODE_DIRECTORY = os.path.abspath(HERE + '/../')
SHARED_DIRECTORY = os.path.abspath(CODE_DIRECTORY + '/shared/')
PROJECT_DIRECTORY = os.path.abspath(HERE + '/../../')
sys.path.append(CODE_DIRECTORY)
sys.path.append(SHARED_DIRECTORY)

# nonstandard imports
from exponential_fitting import exponential_decay, rebin_data
from wio.file_manager import ensure_dir_exists
from fit_plate_timeseries import parse_plate_timeseries_txt_file, parse_plate_timeseries_json_file
from plot_fit_params import ex_id_to_age


TIME_SERIES_DIR = HERE + '/../Data/Time-Series/'
FITS_DIR = HERE + '/../Data/Fits/'
RESULTS_DIR = HERE + '/../Results/'

def plot_plate(ptimes, pmedian, pq1st, pq3rd, wtimes=[], wvalues=[], fit_times=[], fit=[], ground=True, ylabel=''):
    fig = plt.figure()
    ymax_list = []
    scaling_factor = 1.0 / (60)
    if ptimes and pmedian: 
        ptimes = np.array(ptimes) * scaling_factor
        plt.plot(ptimes, pmedian, color='blue')
        ymax_list.append(max(pmedian))
        if pq1st and pq3rd:
            plt.plot(ptimes, pq1st, color='blue', alpha=0.3)
            plt.plot(ptimes, pq3rd, color='blue', alpha=0.3)
            plt.fill_between(ptimes, pq1st, pq3rd, color='blue', alpha=0.2)
            ymax_list.append(max(pq3rd))
    if len(fit_times) and len(fit):
        fit_times = np.array(fit_times) * scaling_factor
        plt.plot(fit_times, fit, color='red', lw=2)
        ymax_list.append(max(fit)) 
    if wtimes and wvalues:
        wtimes = np.array(wtimes) * scaling_factor
        plt.plot(wtimes, wvalues, color='green')
        ymax_list.append(max(wvalues))
    if ground:
        plt.ylim([0, max(ymax_list)])
    plt.xlim([0, max(ptimes)])
    plt.xlabel('time (m)')
    plt.ylabel(ylabel)

    plt.show()

def get_plate_timeseries(ex_id, dataset):
    times, median, q1st, q3rd = [], [], [], []
    plate_file = '{path}/{dset}/{eID}.txt'.format(path=TIME_SERIES_DIR.rstrip('/'), 
                                                  dset=dataset, eID=ex_id)
    if os.path.isfile(plate_file):
        times, data = parse_plate_timeseries_txt_file(plate_file)
    else:
        search = '{path}/{dset}/{eID}*'.format(path=TIME_SERIES_DIR.rstrip('/'), dset=dataset, eID=ex_id)
        print search
        json_file = glob.glob(search)[0]
        print json_file
        times, data = parse_plate_timeseries_json_file(json_file)
    x, bins, n = rebin_data(times, bins=data, t_step=10)
    times = x
    median = [np.median(b) for b in bins]
    q1st = [stats.scoreatpercentile(b, 25) for b in bins]
    q3rd = [stats.scoreatpercentile(b, 75) for b in bins]
    return times, median, q1st, q3rd

def get_plate_fit(ex_id, dataset):
    # get fit function
    fit_times, fit = [], []
    dfits = '{path}/{dset}.json'.format(path=FITS_DIR.rstrip('/'), dset=dataset)
    if os.path.isfile(dfits):
        all_fits = json.load(open(dfits, 'r'))
        print len(all_fits)
        print ex_id
        p = all_fits.get(ex_id, {}).get('params', [])
        if p:
            fit_times = np.linspace(0.0, 3600.0, 3600)            
            fit = exponential_decay(p, fit_times)
        print p
    return fit_times, fit

def get_worm_timeseries(blob_id, dataset):

    return [], []

def print_attributes(ex_id):
    age = ex_id_to_age(ex_id)
    print ex_id, 'is', age, 'hours old'

if __name__ == '__main__':
    dataset = 'N2_aging-centroid_speed'
    #dataset = 'N2_aging-smooth_length'
    #dataset = 'N2_aging-curvature_all_bl'
    eID = '20130318_165649'
    #eID = '20130318_153749'
    blob_id = '20130318_165649_XYZ'

    y_label = dataset.split('-')[-1]
    print_attributes(ex_id=eID)
    ptimes, pmedian, pq1st, pq3rd = get_plate_timeseries(ex_id=eID, dataset=dataset)
    print 'median of medians=', np.median(pmedian)
    wtimes, wvalues = get_worm_timeseries(blob_id, dataset)
    fit_times, fit = get_plate_fit(ex_id=eID, dataset=dataset)
    plot_plate(ptimes, pmedian, pq1st, pq3rd, 
               wtimes, wvalues, fit_times, fit,
               ylabel=y_label)
