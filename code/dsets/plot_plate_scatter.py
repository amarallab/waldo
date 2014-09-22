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

from six.moves import zip

# standard imports
import os
import sys
import matplotlib.pyplot as plt
from mpltools import style
import pandas as pd

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(os.path.join(HERE, '..'))
SHARED_DIR = os.path.abspath(os.path.join(CODE_DIR, 'shared'))
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.plate_utilities import read_dset_summary
from metrics.measurement_switchboard import FULL_SET, STANDARD_MEASUREMENTS

# global settings
style.use('ggplot')

def plot_timeseries(ax, labels, xs, ys, bars):
    colors =['blue', 'red', 'green', 'black', 'orange']
    for c, l in zip(colors, labels):
        x = xs[l]
        y = ys[l]
        set_bars = bars[l]
        for xi, b in zip(x, set_bars):
            b1, b2 = b
            ax.plot([xi, xi], [b1, b2], lw=0.5, color=c)
        ax.plot(x,y, color=c, lw=0, marker='o', alpha=1.0, label=l)

def plot_plate_scatter(ax, dataset, data_type, show_mean, age_range=[0,1000000], labels='all'):

    data = read_dset_summary(dataset=dataset, data_type=data_type)
    if len(data) == 0:
        return
    df = pd.DataFrame(data)
    # calculate two kinds of error bars
    df['std1'], df['std2'] = zip(*[[m-s, m+s] for (m,s) in zip(df['mean'], df['std'])])
    df['q1'], df['median'], df['q3'] = zip(*df['quartiles'])

    print df.head()

    if labels == 'all':
        labels = list(set(df['labels']))

    if show_mean:
        primary = 'mean'
        bar_l, bar_u = ('std1', 'std2')
    else:
        primary = 'median'
        bar_l, bar_u = ('q1', 'q3')

    set_names, set_xs, set_ys, set_bars = [], {}, {}, {}
    for l, grp in df.groupby('labels'):
        # do not do anything if a point is not in labels
        if l in labels:
            set_names.append(l)
            set_xs[l] = grp['hours']
            set_ys[l] = grp[primary]
            set_bars[l] = list(zip(grp[bar_l], grp[bar_u]))

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
    fig, axes = plt.subplots(N, 1, sharex=True)
    for ax, data_type in zip(axes, data_types):
        plot_plate_scatter(ax, dataset, data_type, show_mean,
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

    dtypes = FULL_SET[:-2]

    if dataset == 'N2_aging':
        N_days = 9
        labels = ['set A', 'set B']

    # labels for disease_models
    if dataset == 'disease_models':
        labels = [u'NQ19', u'NQ67', u'MQ0', u'N2', u'MQ40', u'NQ40', u'MQ35']
        #labels = [u'N2', u'MQ35', u'MQ40']
        #labels = [u'N2', u'MQ0']
        #labels = [u'N2', u'NQ40', u'NQ67']
        labels = [u'N2', u'NQ19', u'NQ67']
        #labels = ['N2', 'NQ67']

    sets = plot_dset(dataset=dataset, show_mean=show_mean, age_range=age_range, data_types=dtypes, labels=labels)

