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

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIRECTORY = os.path.abspath(HERE + '/../../')
sys.path.append(PROJECT_DIRECTORY)

# nonstandard imports
from plate_utilities import read_dset_summary


def plot_timeseries(labels, xs, ys, bars):
    colors =['blue', 'red', 'green', 'black', 'orange']
    for c, l in zip(colors, labels):
        print l
        x = xs[l]
        y = ys[l]
        set_bars = bars[l]        
        for xi, b in izip(x, set_bars):
            b1, b2 = b
            plt.plot([xi, xi], [b1, b2], lw=0.5, color=c)
        plt.plot(x,y, color=c, lw=0, marker='o', alpha=1.0, label=l)
    plt.legend()
    plt.show()                                        

def plot_dset(dataset, data_type, plot_attribute, age_range=[0,1000000], split_by_sublabel=False):

    data = read_dset_summary(dataset=dataset, data_type=data_type)
    print data.keys()

    
    labels = data['labels']
    sublabels = data['sub']
    ex_ids = data['ex_ids']
    ages = data['ages']

    # combine labe
    if split_by_sublabel:
        #labels = ['{l}-{s}'.format(l=l, s=s) for (l,s) in zip(labels, sublabels)]
        labels = sublabels

    print 'labels: {N}'.format(N=len(labels))
    print 'ex_ids: {N}'.format(N=len(ex_ids))
    print 'ages: {N}'.format(N=len(ages))

    # get graph ready if mean or median 
    if plot_attribute == 'mean':
        primary = data.get('mean', [])
        stds = data.get('std', [])
        bars = [[m-s, m+s] for (m,s) in izip(primary, stds)]
    if plot_attribute == 'median':
        quantiles = data.get('quartiles', [[], [], []])
        q1, primary, q3 = zip(*quantiles)
        bars = zip(q1, q3)
    print 'primary: {N}'.format(N=len(primary))
    print 'bars: {N}'.format(N=len(bars))

    set_names = []        
    set_xs = {}
    set_bars = {}
    set_ys = {}
    for (l, eID, age, m, bar) in izip(labels, ex_ids, ages, primary, bars):
        if l not in set_names:
            print l
            set_names.append(l)
            set_xs[l] = []
            set_bars[l] =[]
            set_ys[l] = []

        set_xs[l].append(age)
        set_bars[l].append(bar)
        set_ys[l].append(m)

    plot_data = {'labels':sorted(set_names),
                 'xs':set_xs, 'ys':set_ys,
                 'bars':set_bars}

    plot_timeseries(**plot_data)
    return plot_data

    '''
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
    '''

if __name__ == '__main__':
    dataset = 'disease_models'
    #data_type = 'cent_speed_bl'
    data_type = 'length_mm'
    #data_type = 'curve_bl'
    
    # data source toggles
    # plot toggles
    plot_attribute = 'median'
    #plot_attribute = 'mean'

    # optional toggles
    age_range=[0,1000000]
    #age_range=[40,210]
    #age_range=[40,260]
    
    sets = plot_dset(dataset=dataset, data_type=data_type, plot_attribute=plot_attribute, age_range=age_range)
    
