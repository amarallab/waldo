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
from exponential_fitting import fit_exponential_decay_robustly, exponential_decay

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIRECTORY = os.path.abspath(HERE + '/../../')
sys.path.append(PROJECT_DIRECTORY)

# nonstandard imports
from exponential_fitting import fit_exponential_decay_robustly, rebin_data, exponential_decay


BLACKLIST = [] #['20130321_113015', '20130325_152702', '20130420_155856']

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

def plot_params(dsets, labels,age_range=[]):
    fig1 = plt.figure()
    colors =['blue', 'red']
    for dset, label, color in zip(dsets, labels, colors):
        num_params = len(dset[0]) - 2
        ex_ids, ages, params = [], [], []
        for datum in dset:
            ex_ids.append(datum[0])
            ages.append(datum[1])
            params.append(datum[2:])
        params = zip(*params)
        ax1 = plt.subplot(num_params, 1, 1)
        plt.plot(ages, params[0], color=color, marker='o', lw=0, label=label)        
        plt.ylabel('A')
        plt.title('y(t) = A exp( -t / tau) + C')
        #plt.yscale('log')
        if num_params > 1:
            ax2 = plt.subplot(num_params, 1, 2, sharex=ax1)
            y = 1.0 / np.array(params[1])
            plt.plot(ages, y, color=color, marker='o', lw=0, label=label)
            plt.yscale('log')
            plt.ylabel('tau')
        if num_params > 2:
            ax3 = plt.subplot(num_params, 1, 3, sharex=ax1)
            plt.plot(ages, params[2], color=color, marker='o', lw=0, label=label)
            plt.ylabel('C')            
    if age_range:
        plt.xlim(age_range)

    plt.xlabel('age (hours)')
    plt.legend()
    plt.show()

def plot_quantiles(dsets, labels, fit_function=False, age_range=[]):
    fig1 = plt.figure()
    colors =['blue', 'red']
    ax1 = plt.subplot(1, 1, 1)
    for dset, label, color in zip(dsets, labels, colors):
        ex_ids, ages, q1s, medians, q3s = zip(*dset)
        for x, q1, q3 in izip(ages, q1s, q3s):
            plt.plot([x, x], [q1, q3], lw=0.5, color=color)
        plt.plot(ages, medians, color=color, marker='o', lw=0, label=label)
        if fit_function:
            p, l, err = fit_exponential_decay_robustly(ages, medians)
            xfit = np.linspace(min(ages), max(ages), 3000)
            yfit = exponential_decay(p, xfit)
            plt.plot(xfit, yfit, color=color)
    if age_range:
        plt.xlim(age_range)
    plt.xlabel('age (hours)')
    plt.legend()
    plt.show()

def plot_means(dsets, labels, fit_function=False, age_range=[]):
    fig1 = plt.figure()
    colors =['blue', 'red']
    ax1 = plt.subplot(1, 1, 1)
    for dset, label, color in zip(dsets, labels, colors):
        ex_ids, ages, means, stds = zip(*dset)
        for x, mean, std in izip(ages, means, stds):
            plt.plot([x, x], [mean-std, mean+std], lw=0.5, color=color)
        plt.plot(ages, means, color=color, marker='o', lw=0, label=label)
        if fit_function:
            p, l, err = fit_exponential_decay_robustly(ages, means)
            xfit = np.linspace(min(ages), max(ages), 3000)
            yfit = exponential_decay(p, xfit)
            plt.plot(xfit, yfit, color=color)

    if age_range:
        plt.xlim(age_range)
    plt.xlabel('age (hours)')
    plt.legend()
    plt.show()

def ex_id_to_age(ex_id):
    ''' returns the age of the worms (hours) at the start of
    a plate in the N2_aging dataset.
    '''
    platetime1 = datetime.datetime(2013, 03, 16, 11, 00)
    platetime2 = datetime.datetime(2013, 04, 06, 11, 00)
    if int(ex_id.split('_')[0]) < 20130407:
        pt = platetime1
    else:
        pt = platetime2    
    return (ex_id_to_datetime(ex_id) - pt).total_seconds()/3600.

    
def parse_datasets(data_dict, plot_attribute='params', age_range=[0,1000000]):
    # parses the N2_aging datasets into two seperate runs.
    dset1, dset2 = [], []
    for ex_id in sorted(data_dict)[:]:
        if int(ex_id.split('_')[0]) < 20130407:
            dset = dset1
        else:
            dset = dset2
        age = ex_id_to_age(ex_id)
        d = data_dict[ex_id]
        min_age, max_age = age_range       
        if d and (min_age <= age <= max_age):
            N = d['mean_N']
            if plot_attribute == 'means':
                params = [d['mean'], d['std']]
            else:
                params = d[plot_attribute]

            if len(params) > 0 and  N >900:
                #print ex_id, age, N
                datum = [ex_id, age] + list(params)
                dset.append(datum)
    return dset1, dset2

def plot_multiple_fits(dsets, labels, age_range=[46, 60]):
    fig1 = plt.figure()
    colors =['blue', 'red']
    x_fit = np.linspace(0.0, 3600.0, 3600)            
    ax1 = plt.subplot(1, 1, 1)    
    for dset, label, color in zip(dsets, labels, colors):
        ex_ids, ages, params = [], [], []        
        for datum in dset:
            eID, age, p = datum[0], datum[1], datum[2:]
            if age_range[0] <= age <= age_range[1]:
                ex_ids.append(eID)
                ages.append(age)
                params.append(p)
        mean_p, median_p = [], []
        for i in zip(*params):
            mean_p.append(np.mean(i))
            median_p.append(np.median(i))
        
        for age, p in zip(ages, params):            
                plt.plot(x_fit, exponential_decay(p, x_fit),
                         color=color, alpha=0.9, lw=0.5)
        plt.plot(x_fit, exponential_decay(mean_p, x_fit),
                 color=color, label=label, lw=2)
                
    #plt.xlabel('age (hours)')
    plt.legend()
    plt.show()


def main_plot_stationary(dfile, plot_attribute, fit=False, age_range=[0,1000000]):
    print dfile, os.path.isfile(dfile)
    dsets = parse_datasets(json.load(open(dfile, 'r')), plot_attribute,
                           age_range=age_range)
    print len(dsets)

    #plot_params(dsets=dsets, labels=['2013-03-18', '2013-04-08'])
    labels = ['N2 set A', 'N2 set B']

    if plot_attribute == 'params':
        plot_params(dsets=dsets, labels=labels, age_range=age_range)
    elif plot_attribute == 'quartiles':
        plot_quantiles(dsets=dsets, labels=labels, fit_function=fit, age_range=age_range)
    elif plot_attribute == 'means':
        plot_means(dsets=dsets, labels=labels, fit_function=fit, age_range=age_range)
       
def main_plot_multiple_fits(dfile, age_range):
    plot_attribute = 'params'
    
    dsets = parse_datasets(json.load(open(dfile, 'r')), plot_attribute,
                           age_range=age_range)
    for i, dset in enumerate(dsets):
        print i, len(dset)
    plot_multiple_fits(dsets=dsets, labels=['2013-03-18', '2013-04-08'])
    
if __name__ == '__main__':
    dataset = 'disease_models'
    data_type = 'cent_speed_bl'
    
    # data source toggles

    # plot toggles
    plot_attribute = 'params'
    #plot_attribute = 'quartiles'
    #plot_attribute = 'means'

    # optional toggles
    age_range=[0,1000000]
    #age_range=[40,210]
    #age_range=[40,260]
    #age_range=[40,70] 
    fit = False
    
    main_plot_stationary(dfile, plot_attribute, fit=False, age_range=age_range)
    #main_plot_multiple_fits(dfile, age_range)
