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
from mpltools import style
import pandas as pd

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(os.path.join(HERE, '..'))
SHARED_DIR = os.path.abspath(os.path.join(CODE_DIR, 'shared'))
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.plate_utilities import read_dset_summary
from wormmetrics.measurement_switchboard import FULL_SET, STANDARD_MEASUREMENTS

# global settings
style.use('ggplot')

def plot_timeseries(ax, labels, xs, ys, bars):
    colors =['blue', 'red', 'green', 'black', 'orange']
    for c, l in zip(colors, labels):
        print l
        x = xs[l]
        y = ys[l]
        set_bars = bars[l]        
        for xi, b in izip(x, set_bars):
            b1, b2 = b
            ax.plot([xi, xi], [b1, b2], lw=0.5, color=c)
        ax.plot(x,y, color=c, lw=0, marker='o', alpha=1.0, label=l)


def plot_data_type_plate_scatter2(ax, dataset, data_type, show_mean, age_range=[0,1000000]):

    data = read_dset_summary(dataset=dataset, data_type=data_type)
    df = pd.DataFrame()
    df['labels'] = data['labels']
    df['sub'] = data['sub']
    df['ex_ids'] = data['ex_ids']
    df['hours'] = data['hours']
    df['mean'] = data.get('mean', [])
    df['stds'] = data.get('std', [])
    '''
    df = pd.DataFrame(data)
    df['labels'] = data['labels']
    df['sub'] = data['sub']
    df['ex_ids'] = data['ex_ids']
    df['hours'] = data['hours']
    df['mean'] = data.get('mean', [])
    df['stds'] = data.get('std', [])
    '''
    # calculate bars
    std1, std2 = zip(*[[m-s, m+s] for (m,s) in izip(df['mean'], df['stds'])])
    df['std1'], df['std2'] = std1, std2        
    q1, med, q3 = zip(*data.get('quartiles', [[], [], []]))    
    df['q1'] = q1
    df['median'] = med
    df['q3'] = q3
        
       
    if show_mean:
        primary = 'mean'
        bar_l, bar_u = ('std1', 'std2')
    else:
        primary = 'median'
        bar_l, bar_u = ('q1', 'q2')

    set_names, set_xs, set_ys, set_bars = [], {}, {}, {}
    for l, grp in df.groupby('labels'):
        # do not do anything if a point is not in labels
        if l in labels:
            set_names.append(l)
            set_xs[l] = grp['hours']
            set_ys[l] = grp[primary]
            set_bars[l] = zip(grp[bar_l], grp[bar_u])
            
    plot_data = {'ax':ax,
                 'labels':sorted(set_names),
                 'xs':set_xs, 'ys':set_ys,
                 'bars':set_bars}

    plot_timeseries(**plot_data)
    ax.set_ylabel(data_type)
    return plot_data
        
        
        
def plot_data_type_plate_scatter(ax, dataset, data_type, show_mean, age_range=[0,1000000], split_by_sublabel=False):
    data = read_dset_summary(dataset=dataset, data_type=data_type)
    print data.keys()

    all_labels = data['labels']
    if labels == 'all':
        labels = list(set(all_labels))
    
    for l in labels:
        assert l in all_labels
    
    sublabels = data['sub']
    ex_ids = data['ex_ids']
    ages = data['hours']

    # combine labe
    #if split_by_sublabel:
    #    labels = ['{l}-{s}'.format(l=l, s=s) for (l,s) in zip(labels, sublabels)]
            
    print 'labels: {N}'.format(N=len(labels))
    print 'ex_ids: {N}'.format(N=len(ex_ids))
    print 'ages: {N}'.format(N=len(ages))

    # get graph ready if mean or median 
    #if show_mean == 'mean':
    if show_mean:
        primary = data.get('mean', [])
        stds = data.get('std', [])
        bars = [[m-s, m+s] for (m,s) in izip(primary, stds)]
    else:
        quantiles = data.get('quartiles', [[], [], []])
        q1, primary, q3 = zip(*quantiles)
        bars = zip(q1, q3)
    print 'primary: {N}'.format(N=len(primary))
    print 'bars: {N}'.format(N=len(bars))

    set_names = []        
    set_xs = {}
    set_bars = {}
    set_ys = {}
    for (l, eID, age, m, bar) in izip(all_labels, ex_ids, ages, primary, bars):
        # do not do anything if a point is not in labels
        if l not in labels:
            continue
        
        if l not in set_names:
            set_names.append(l)
            set_xs[l] = []
            set_bars[l] =[]
            set_ys[l] = []

        set_xs[l].append(age)
        set_bars[l].append(bar)
        set_ys[l].append(m)

    plot_data = {'ax':ax,
                 'labels':sorted(set_names),
                 'xs':set_xs, 'ys':set_ys,
                 'bars':set_bars}

    plot_timeseries(**plot_data)
    ax.set_ylabel(data_type)
    return plot_data


def plot_dset(dataset, data_types=FULL_SET, show_mean=False,
              age_range=[0,1000000], labels='all'):

    N = len(data_types)
    fig, axes = plt.subplots(N, 1)
    for ax, data_type in zip(axes, data_types):
        plot_data_type_plate_scatter(ax, dataset, data_type, show_mean, 
                                     age_range=age_range, labels=labels)
    plt.legend()
    plt.show()                                        


if __name__ == '__main__':
    dataset = 'disease_models'
    dataset = 'N2_aging'
    #data_type = 'length_mm'
    #data_type = 'curve_w'
    
    # data source toggles
    # plot toggles
    show_mean = False
    #show_mean = 'mean'

    # optional toggles
    age_range=[0,1000000]
    #age_range=[40,210]
    #age_range=[40,260]    
    sets = plot_dset(dataset=dataset, show_mean=show_mean, age_range=age_range)
    
