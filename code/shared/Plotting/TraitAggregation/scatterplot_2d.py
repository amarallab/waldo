#!/usr/bin/env python

'''
Filename: scatterplot_2d.py
Description: 
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
from itertools import izip
import pylab as pl
import numpy as np

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../../'
assert os.path.exists(project_directory), 'project directory not found'
sys.path.append(project_directory)

# nonstandard imports
from Shared.Code.Settings.data_settings import plotting_locations
import Shared.Code.Plotting.utilities as ut

def scatterplot(data_sets, labels, x_label='', y_label='', savename='', num_bins=20):
    # toggles
    set_colors = ['blue', 'red', 'green', 'black', 'yellow', 'cyan', 'magenta']

    # definitions for the axes
    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.6
    bottom_h = left_h = left + width + 0.02

    # start with a rectangular Figure
    pl.figure(1, figsize=(8, 8))

    # define all plots and set locations
    axScatter = pl.axes([left, bottom, width, height])
    axHistx = pl.axes([left, bottom_h, width, 0.2])
    axHisty = pl.axes([left_h, bottom, 0.2, height])

    # loop through datasets and
    for i, dset in enumerate(data_sets):
        if len(dset) ==0:
            continue
        x, y = dset
        if not x or not y:
            nx, ny = [], []
            continue
        set_color = set_colors[i]
        axScatter.scatter(x, y, color=set_color, alpha=0.3, label=labels[i])

        nx, binsx = np.histogram(x, num_bins, range=[np.min(x), np.max(x)], normed=True)
        bincenters = list(0.5 * (binsx[1:] + binsx[:-1]))
        #bincenters.append(binsx[-1])
        #nx = list(nx).append(0)
        axHistx.fill_between(bincenters, 0, nx, facecolor=set_color, alpha=0.3)
        axHistx.plot(bincenters, nx, color=set_color, alpha=0.7)
        #axHistx.tick_top()
        #axHistx.set_ticks_position('top')

        ny, binsy = np.histogram(y, num_bins, range=[np.min(y), np.max(y)], normed=True)
        bincenters = 0.5 * (binsy[1:] + binsy[:-1])
        #axHisty.fill_between(ny, bincenters, np.min(ny), facecolor=set_color, alpha=0.3)

        new_bincenters = [binsy[0]]
        new_ny = [0]
        for N, bc in izip(ny, bincenters):
            new_bincenters.append(bc)
            new_ny.append(N)
        new_bincenters.append(binsy[-1])
        new_ny.append(0)

        #ny = list(ny).append(0)
        #axHisty.fill_between(new_ny, new_bincenters, np.min(new_bincenters), facecolor=set_color, alpha=0.3)
        axHisty.fill_between(new_ny, new_bincenters, axScatter.get_ylim()[-1], facecolor=set_color, alpha=0.2)

        #axHisty.fill_between(ny, 0, bincenters, facecolor=set_color, alpha=0.3)
        #axHisty.fill_between(bincenters, 0, ny, facecolor=set_color, alpha=0.3)
        axHisty.plot(ny, bincenters, color=set_color, alpha=0.2)
        axHisty.yaxis.tick_right()

    axScatter.legend(loc=2)
    axScatter.xaxis.set_label_text(x_label)
    axScatter.yaxis.set_label_text(y_label)

    if len(nx) > 0:
        axHistx.set_xlim(axScatter.get_xlim())
        axHistx.set_ylim([0, np.max(nx)])
    if len(ny) > 0:
        axHisty.set_ylim(axScatter.get_ylim())
        axHisty.set_xlim([0, np.max(ny)])
    if not savename:
        pl.show()
    else:
        pl.savefig(savename)
    pl.clf()

# I think this is depreciated
'''
def multi_query_plot(queries, labels, data_type1, data_type2, metric1, metric2, savename):
    """
    :param queries:
    :param data_type1:
    :param data_type2:
    """

    key1 = '_'.join([data_type1, metric1])  #data_type1 + '_mean'
    key2 = '_'.join([data_type2, metric2]) # data_type2 + '_mean'
    keys = [key1, key2]

    dsets = []
    for query in queries:
        assert type(query) == dict
        dsets.append(ut.get_multiple_matched_results(query, keys))

    # another way of doing the for loop
    dsets = [ut.get_multiple_matched_results(q, keys)
              for q in queries]
    scatterplot(data_sets=dsets, labels=labels, x_label=key1, y_label=key2, savename=savename)


def multi_list_plot(id_lists, labels, data_type1, data_type2, plot_type, savename):
    """

    :param queries:
    :param data_type1:
    :param data_type2:
    :param plot_type:
    """
    key1 = data_type1 + '_mean'
    key2 = data_type2 + '_mean'
    #key1 = 'pc1_median'
    #key2 = 'pc2_median'
    keys = [key1, key2]

    dsets = []
    for blob_ids in id_lists:
        assert type(blob_ids) == list
        dsets.append(ut.get_multiple_matched_results_for_ids(blob_ids=blob_ids, keys=keys))
    scatterplot(dsets, labels, key1, key2, savename)

def id_list_preprocess(data_type1, data_type2, instructions=[]):
    assert len(instructions) > 0
    labels, id_lists = zip(*instructions)
    plot_type = 'mean_point'
    save_name = '%s2_%s_%s.png' % (plotting_locations['TraitAggregation'], data_type1, data_type2)
    multi_list_plot(id_lists, labels, data_type1, data_type2, plot_type, save_name)
'''
