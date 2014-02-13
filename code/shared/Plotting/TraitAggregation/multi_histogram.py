
# standard imports
import os
import sys
import scipy.stats as stats
from scipy.stats import scoreatpercentile
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from random import random

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../../'
assert os.path.exists(project_directory), 'project directory not found'
sys.path.append(project_directory)

# nonstandard imports
#from Shared.Code.Settings.data_settings import default_datatypes
from Shared.Code.Plotting.utilities import pull_datasets_for_data_type

def table_histogram_set(datasets, labels, x_label=None, num_bins=50, savename=None, guidelines=True, title=''):
    datasets = datasets[::-1]
    labels = labels[::-1]

    fig = plt.figure(1, figsize=(8, 8))
    if title:
        plt.title(title)
    ax = fig.add_subplot(111)
    # toggles to set how much of data to include in range.
    #total_max = max([max(dset) for dset in datasets])
    #total_min = min([min(dset) for dset in datasets])
    total_max = max([scoreatpercentile(dset, 98) for dset in datasets if len(dset) > 3])
    total_min = min([scoreatpercentile(dset, 02) for dset in datasets if len(dset) > 3])
    scaling_spread = 0.6

    if guidelines:
        median, guide_left, guide_right = [], [], []
        for i, dset in enumerate(datasets):
            median.append((scoreatpercentile(dset, 50), i))
            median.append((scoreatpercentile(dset, 50), i+1))
            guide_left.append((scoreatpercentile(dset, 25), i))
            guide_left.append((scoreatpercentile(dset, 25), i+1))
            guide_right.append((scoreatpercentile(dset, 75), i))
            guide_right.append((scoreatpercentile(dset, 75), i+1))
        ax.plot(*zip(*median), color='black', alpha=0.2)
        ax.plot(*zip(*guide_left), color='black', alpha=0.2)
        ax.plot(*zip(*guide_right), color='black', alpha=0.2)

    sf = 10000.0
    for dset in datasets:
        if len(dset) == 0:
            continue
        n, _ = np.histogram(dset, num_bins, range=[total_min, total_max]) 
        sf = min([sf, (0.8 * len(dset)/ max(n))])

    for i, dset in enumerate(datasets):
        ypos = i
        ax.text(total_min, ypos, labels[i], horizontalalignment='left', verticalalignment='top')
        ax.text(total_max, ypos, 'N={n}'.format(n=len(dset)), horizontalalignment='right', verticalalignment='top')
        if len(dset) == 0:
            continue
        #nx, binsx = np.histogram(dset, num_bins, range=[min(dset), max(dset)], normed=True)
        n, bins = np.histogram(dset, num_bins, range=[total_min, total_max])
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        N = float(len(dset))
        y = [(sf * j / N) + ypos for j in n]
        ax.plot(bincenters, y, color='black')
        ax.plot([total_min, total_max], [ypos, ypos], color='black')
        ax.fill_between(bincenters, y, [ypos for j in n], facecolor='blue', alpha=0.2)


    ax.plot([total_min, total_max], [-0.2, -0.2], color='white')
    if x_label:
        ax.set_xlabel(x_label)
    if savename:
        plt.savefig(savename)
    else:
        plt.show()
    plt.clf()

def sideways_hist_plot(datasets, labels, x_label=None, num_bins=50, savename=None, guidelines=True, title=''):
    #datasets = datasets[::-1]
    #labels = labels[::-1]

    fig = plt.figure(1, figsize=(8, 8))
    if title:
        plt.title(title)
    ax = fig.add_subplot(111)
    # toggles to set how much of data to include in range.
    #total_max = max([max(dset) for dset in datasets])
    #total_min = min([min(dset) for dset in datasets])
    total_max = max([scoreatpercentile(dset, 98) for dset in datasets if len(dset) > 3])
    total_min = min([scoreatpercentile(dset, 02) for dset in datasets if len(dset) > 3])
    scaling_spread = 0.6

    sf = 10000.0
    for dset in datasets:
        if len(dset) == 0:
            continue
        n, _ = np.histogram(dset, num_bins, range=[total_min, total_max]) 
        sf = min([sf, (0.8 * len(dset)/ max(n))])

    for i, dset in enumerate(datasets):
        xpos = i
        if len(dset) == 0:
            continue
        #nx, binsx = np.histogram(dset, num_bins, range=[min(dset), max(dset)], normed=True)
        n, bins = np.histogram(dset, num_bins, range=[total_min, total_max])        
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        N = float(len(dset))
        x = [(sf * j / N) + xpos for j in n] + [xpos]
        ax.plot(x ,bincenters, color='black')
        ax.plot([xpos, xpos], [total_min, total_max], color='black')
        #ax.fill_between(bincenters, x, [xpos for j in n], facecolor='blue', alpha=0.2)
        ax.fill_between(x, bincenters, [xpos for _ in bincenters], facecolor='blue', alpha=0.2)
        

    #ax.plot([total_min, total_max], [-0.2, -0.2], color='white')
    if x_label:
        ax.set_xlabel(x_label)
    if savename:
        plt.savefig(savename)
    else:
        plt.show()
    plt.clf()

"""
def plot_multiworm_hists(query, num_bins):
    for data_type in default_datatypes[:]:
        datasets, all_data = pull_datasets_for_data_type(query, data_type)

        if len(all_data) > 100:
            mind = min(all_data)
            maxd = stats.scoreatpercentile(all_data, 98)
            '''
            savename = 'loghist_numbins' + str(num_bins) + '_' + data_type +  '.png'
            print savename
            print len(datasets), 'datasets'
            pl.figure()
            for dset in datasets:
                n, bins = np.histogram(dset, num_bins, range=[mind,maxd] , normed=True)
                bincenters = 0.5*(bins[1:]+bins[:-1])
                pl.plot(bincenters, n, color='black', alpha=0.2)
                #pl.hist(dset, bins=np.logspace(mind, maxd, num_bins), alpha = 0.2)
                #pl.hist(dset, bins=np.logspace(.1, 1.0, num_bins))

            np.histogram(all_data, num_bins, range=[mind,maxd], normed=True)
            bincenters = 0.5*(bins[1:]+bins[:-1])
            pl.plot(bincenters, n, color='red', lw=2)
            pl.xlabel(data_type)
            pl.xscale('log')
            pl.savefig(savename)
            '''
            savename = 'linhist_numbins' + str(num_bins) + '_' + data_type +  '.png'
            print savename
            print len(datasets), 'datasets'
            pl.figure()
            for dset in datasets:
                n, bins = np.histogram(dset, num_bins, range=[mind,maxd] , normed=True)
                bincenters = 0.5*(bins[1:]+bins[:-1])
                pl.plot(bincenters, n, color='black', alpha=0.2)
                #    pl.hist(dset, bins=np.logspace(.1, maxd, num_bins), alpha = 0.2)
            np.histogram(all_data, num_bins, range=[mind,maxd], normed=True)
            bincenters = 0.5*(bins[1:]+bins[:-1])
            pl.plot(bincenters, n, color='red', lw=2)
            #pl.hist(all_data, bins=np.logspace(.1, 500, num_bins))
            pl.xlabel(data_type)
            #pl.xlim([mind, stats.scoreatpercentile(all_data, 98)])
            pl.savefig(savename)
"""
if __name__ == '__main__':
    # toggles
    #query = {'strain': 'N2', 'duration': {'$gt':60}}
    #num_bins = 25
    #plot_multiworm_hists(query, num_bins)

    set_num = 15
    datasets = [np.random.randn(100*i+100) for i in xrange(set_num)]
    labels = ['type {}'.format(i) for i in xrange(set_num)]
    table_histogram_set(datasets=datasets, labels=labels)
