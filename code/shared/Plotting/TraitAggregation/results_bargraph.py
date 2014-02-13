# standard imports
import os
import sys
import math
import random
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile

'''
# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../../'
assert os.path.exists(project_directory), 'project directory not found'
sys.path.append(project_directory)

shared_code_directory = project_directory + 'Shared/Code'
assert os.path.exists(shared_code_directory), 'Shared/Code directory not found'
sys.path.append(shared_code_directory)
'''
'''
def table_boxplot2(labels, datasets, savename='', x_label='value'):
    """ make the plot.
    :param labels:
    :param datasets:
    :param make_boxplot:
    :param x_label:
    :param scatter:
    """
    # toggles
    graph_type = 'bar'
    bargraph = False
    scatter = 0.3
    alpha = 0.4
    y_increment = 2

    graph_types = ['bargraph', 'boxplot', 'point', 'point_boxplot']
    assert graph_type in graph_types
    #errorbars = np.std
    errorbars = lambda d: np.std(d) / math.sqrt(len(d))

    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.6
    # start with a rectangular Figure
    pl.figure(1, figsize=(8, 8))
    # define all plots and set locations
    ax = pl.axes([left, bottom, width, height])

    pos = []
    for i, dset in enumerate(datasets):
        label = labels[i]
        ypos = (y_increment * i) + y_increment
        pos.append(ypos)
        if graph_type == 'bargraph':
            y = [ypos for _ in dset]
            x = [np.mean(dset) for _ in dset]
            pl.barh(y, x, xerr=errorbars(dset), align='center', height=0.1, label=label)
        if graph_type == 'point':
            x = datasets[i]
            y = [ypos + random.random() * scatter - 0.5 * scatter for _ in x]
            pl.plot(x, y, marker='.', color='k', ls='0', alpha=alpha)

    pl.yticks(pos, labels)
    pl.title(x_label)
    pl.ylim([0, ((y_increment * len(datasets)) + y_increment / 2)])
    pl.xlabel(x_label)
    if len(savename) == 0:
        pl.show()
    else:
        pl.savefig(savename)
    pl.clf()
'''

def table_boxplot(labels, datasets, savename='', x_label='value'):
    ''' make the plot.
    :param savename:
    :param labels:
    :param datasets:
    :param make_boxplot:
    :param x_label:
    :param scatter:
    '''
    # toggles
    graph_type = ['box']
    scatter = 0.3
    alpha = 0.4
    y_increment = 2

    errorbars = lambda d: np.std(d) / math.sqrt(len(d))

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=(8, 8))
    # define all plots and set locations
    left, width = 0.25, 0.75
    bottom, height = 0.1, 0.75
    #ax = plt.axes([left, bottom, width, height])
    ax = fig.add_subplot(111)
    fontsize = 'small'
    pos = []
    for i, dset in enumerate(datasets):
        label = labels[i]
        ypos = (y_increment * i) + y_increment
        pos.append(ypos)
        if len(dset) < 4:
            continue

        if 'bar' in graph_type:
            y = [ypos for _ in dset]
            ax.barh(y, [np.mean(dset) for _ in dset], xerr=errorbars(dset), align='center', height=0.1, label=label)
        if 'point' in graph_type:
            r = 0.6
            y = [ypos + random.random() * r * scatter - 0.5 * r * scatter for _ in dset]
            ax.plot(dset, y, marker='.', color='k', ls='0', alpha=alpha)
        if 'box' in graph_type:
            low_end, high_end = scoreatpercentile(dset, 10), scoreatpercentile(dset, 90)
            percentile_box = (low_end, high_end - low_end)
            ax.broken_barh([percentile_box], (ypos - (0.5 * scatter), scatter), alpha=0.6)
            ax.plot([min(dset), low_end], [ypos, ypos], color='blue')
            ax.plot([high_end, max(dset)], [ypos, ypos], color='blue')

        if True: # 'mean_line' in graph_type:
            mean = np.mean(dset)
            ax.text(mean, ypos + scatter, '{:.3e}'.format(mean),
                    horizontalalignment='center', fontsize=fontsize)
            ax.broken_barh([(mean - 0.000001, 0.000002)], (ypos - scatter, 2 * scatter))
        if True: # 'mean_line' in graph_type:
            median = np.median(dset)
            ax.text(median, ypos - scatter, '{:.3e}'.format(median),
                    horizontalalignment='center', verticalalignment='top', color='red', fontsize=fontsize)
            ax.broken_barh([(median - 0.000001, 0.000002)], (ypos - scatter, 2 * scatter), color='red')
        if True:
            low_end, high_end = scoreatpercentile(dset, 10), scoreatpercentile(dset, 90)
            ax.text(low_end, ypos-(0.5 * scatter),'{:.3e}'.format(low_end), color='blue',
                    horizontalalignment='center', verticalalignment='top', fontsize=fontsize)
            ax.text(high_end, ypos+(0.5 * scatter),'{:.3e}'.format(high_end), color='blue',
                    horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize)

            # TODO: fix vertical spacing to make sense with amount of whitespace available
            # TODO: Add the color legend that explains the mean, median, and box


    plt.yticks(pos, labels)
    plt.title(x_label)
    plt.ylim([0, ((y_increment * len(datasets)) + y_increment / 2)])
    plt.xlabel(x_label)
    if len(savename) == 0:
        plt.show()
    else:
        plt.savefig(savename)
    plt.clf()


