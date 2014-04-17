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

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(os.path.join(HERE, '..'))
SHARED_DIR = os.path.abspath(os.path.join(CODE_DIR, 'shared'))
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.plate_utilities import organize_plate_metadata, get_plate_files, return_flattened_plate_timeseries, read_dset_summary
from wormmetrics.measurement_switchboard import FULL_SET, STANDARD_MEASUREMENTS


def plot_distributions_by_day(dataset, data_type, labels, ylim, plotname=None):
    #plt.plot(x, y, label=str(len(s)))
    
    y_lim = [0.0, 0.006]
    #N_days = len(dist_by_day)    
    N_days = 5
    fig = plt.figure()
    gs1 = gridspec.GridSpec(1, N_days)
    gs1.update(left=0.1, right=0.9, wspace=0.00)
    ax1 = plt.subplot(gs1[0, 0])
    ax = [plt.subplot(gs1[0, i], sharey=ax1, sharex=ax1) for i in range(1, N_days)]
    ax = [ax1] + ax

    max_x = 0.0
    colors = ['blue', 'red', 'green', 'black']

    #label_settings = {'top': 'off', 'bottom': 'off', 'right': 'off', 'left': 'off',
    #                  'labelbottom': 'off', 'labeltop': 'off', 'labelright': 'off', 'labelleft': 'off'}

    for color, label in zip(colors, labels):
        
        savename = '{ds}-{dt}-{l}.json'.format(ds=dataset, dt=data_type, l=label)
        savename = '{ds}-{dt}-{l}.json'.format(ds=dataset, dt=data_type, l=label)
        print savename, os.path.isfile(savename)

        #dist_by_day = json.load(open(savename, 'r'))
        dist_by_day = read_dset_summary(dataset=dataset, data_type=data_type, 
                                        ID=label, sum_type='dist')

        for day in sorted(dist_by_day):
            data = dist_by_day[day]
            i = int(day) - 1
            # flip axis
            y, x = data['x'] ,data['y']            
            ax[i].fill_between(x, y, alpha=0.5, color=color)
            ax[i].plot(x, y, alpha=0.5, color=color, label=label)
            #plt.tick_params(**label_settings)
            if max_x < max(x):
                max_x = max(x)
    # make x axis ticks relativly sparse
    for a in ax:
        a.xaxis.set_ticks([0.0, 0.001])
    # remove x axis ticklabels from all but first box
    for a in ax[1:]:
        for tick in a.get_xticklabels():
            tick.set_fontsize(0.0)
    # remove y axis ticklabels from all but first and last box
    for a in ax[1:-1]:
        for tick in a.get_yticklabels():
            tick.set_fontsize(0.0)
    # move tick labels to right side for last box.
    ax[-1].yaxis.tick_right()
    print max_x
    #plt.ylim([0.0, 2.0])
    plt.ylim(ylim)
    plt.xlim([0.0, max_x])
    ax[1].legend()
    if plotname:
        plt.savefig(plotname)
    plt.show()
    plt.clf()
                             
if __name__ == '__main__':
    dataset = 'disease_models'
    #preprocess_standard_set_of_distributions(dataset)

    type_index = 1
    standard_set = ['cent_speed_bl', 'length_mm', 'curve_bl']
    xlims = [[0.0, 0.4], [0.0, 2.0], [0.0, 0.006]]

    data_type = standard_set[type_index]
    xlim = xlims[type_index]

    #data_type = 'cent_speed_bl'
    #data_type = 'length_mm'
    #data_type = 'curve_bl'
    #xlim = [0.0, 0.006]
    #xlim = [0.0, 2.0]
    '''
    label = 'all'
    label = 'N2'
    labels = ['N2']
    '''
    labels = [u'NQ19', u'NQ67', u'MQ0', u'N2', u'MQ40', u'NQ40', u'MQ35']
    #for label in labels:
    #    preprocess_distribution_data(dataset, data_type, label, xlim=xlim)

    #labels = [u'NQ19', u'NQ67', u'MQ0', u'N2', u'MQ40', u'NQ40', u'MQ35']
    #labels = [u'N2', u'NQ67']
    #labels = [u'N2', u'MQ40']    
    labels = [u'N2', u'MQ35', u'MQ40']
    labels = [u'N2', u'MQ0']
    labels = [u'N2', u'NQ40', u'NQ67']
    for i, labels in enumerate([[u'N2', u'MQ35', u'MQ40'],
                                [u'N2', u'MQ0'],
                                [u'N2', u'NQ40', u'NQ67']]):

        plotname = '{set}-{dtype}-{i}.png'.format(set=dataset, dtype=data_type, i=i)
        print plotname
        plot_distributions_by_day(dataset, data_type, labels, ylim=xlim, plotname=plotname)
