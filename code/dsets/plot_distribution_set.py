#!/usr/bin/env python

'''
Filename: plot_dataset_distributions.py
Description:
Code for plotting the distributions of aging for different
labels inside the same dataset for multiple different types of measurements.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import numpy as np
import random
from itertools import izip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from mpltools import style

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(os.path.join(HERE, '..'))
SHARED_DIR = os.path.abspath(os.path.join(CODE_DIR, 'shared'))
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.plate_utilities import organize_plate_metadata, get_plate_files, return_flattened_plate_timeseries, read_dset_summary
from wormmetrics.measurement_switchboard import FULL_SET, STANDARD_MEASUREMENTS
from importing.consolidate_dataset import XLIMS

# global settings
style.use('ggplot')


def plot_distribution_row(ax_row, dataset, data_type, labels, ylim, colors=None):
    N_days = len(ax_row)
    max_x = 0.0
    
    #colors = ['blue', 'red', 'green', 'black']
    

    for j, label in enumerate(labels):        
        dist_by_day = read_dset_summary(dataset=dataset, data_type=data_type, 
                                        ID=label, sum_type='dist')
        print label, len(dist_by_day), dist_by_day.keys()

        for day in sorted(dist_by_day):
            data = dist_by_day.get(day, None)
            if data == None:
                print day, 'has no data', label
                continue
            i = int(day) - 1

            if i >= N_days:
                continue

            # flip ax_rowis
            y, x = data['x'] ,data['y']
            if colors != None:
                color = colors[j]
            else:
                color = ax_row[i]._get_lines.color_cycle.next()
                #print color_cycle
                #color = color_cycle()
            ax_row[i].fill_between(x, y, alpha=0.5, color=color)
            ax_row[i].plot(x, y, alpha=0.5, color=color, label=label)

            #plt.tick_params(**label_settings)
            if max_x < max(x):
                max_x = max(x)

    # make x axis ticks relativly sparse
    for a in ax_row:
        a.xaxis.set_ticks([0.0, 0.001])
    # remove x axis ticklabels from all but first box
    for a in ax_row[1:]:
        for tick in a.get_xticklabels():
            tick.set_fontsize(0.0)
    # remove y axis ticklabels from all but first and last box
    for a in ax_row[1:-1]:
        for tick in a.get_yticklabels():
            tick.set_fontsize(0.0)
    # move tick labels to right side for last box.
    ax_row[-1].yaxis.tick_right()
    ax_row[0].set_ylabel(data_type)

    ax_row[0].set_ylim(ylim)
    ax_row[0].set_xlim([0.0, max_x])


def plot_distribution_matrix(dataset, data_types=FULL_SET, labels=['all'], N_days=5, show_legend=True):

    N_rows = len(data_types)
    #fig = plt.figure()
    grid = gridspec.GridSpec(N_rows, N_days)
    grid.update(left=0.2, right=0.9, wspace=0.00)

    axes_rows = []
    for i, data_type  in enumerate(data_types):    
        ax1 = plt.subplot(grid[i, 0])
        ax_row = [plt.subplot(grid[i, j], sharey=ax1, sharex=ax1) for j in range(1, N_days)]
        ax_row = [ax1] + ax_row

        ylim = XLIMS.get(data_type, [0.0, 1.0])
        plot_distribution_row(ax_row, dataset, data_type, labels, ylim)
        axes_rows.append(ax_row)

    ax_row = axes_rows[0]
    ax_row[-1].legend(loc='best', ncol=1)

    #axes_rows[0][1]
    plt.show()
                         
if __name__ == '__main__':
    dataset = 'disease_models'
    #dataset = 'N2_aging'
    #dataset = 'thermo_recovery'


    dtypes = FULL_SET
    #dtypes = ['length_mm']
    show_legend = True

    # defaults
    N_days = 5    
    labels = ['all']

    # labels for N2_aging
    if dataset == 'N2_aging':
        N_days = 9
        labels = ['set A', 'set B']

    # labels for disease_models
    if dataset == 'disease_models':
        #labels = [u'NQ19', u'NQ67', u'MQ0', u'N2', u'MQ40', u'NQ40', u'MQ35']
        #labels = [u'N2', u'MQ35', u'MQ40']
        #labels = [u'N2', u'MQ0']
        labels = [u'N2', u'NQ40', u'NQ67']
        #labels = ['N2', 'NQ67']


    plot_distribution_matrix(dataset, labels=labels, data_types=dtypes, N_days=N_days, show_legend=show_legend)
