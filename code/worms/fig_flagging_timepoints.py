#!/usr/bin/env

# standard library
import sys
import os
from itertools import izip

# third party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_directory)

# waldo
from conf import settings
import database.mongo_support_functions as mongo
from database.mongo_retrieve import mongo_query
from database.mongo_retrieve import pull_data_type_for_blob
from database.mongo_retrieve import timedict_to_list

from experiment_index import Experiment_Attribute_Index
import flags_and_breaks

DATA_DIR = settings.LOGISTICS['filesystem_data']
SAVE_DIR = './../Results/FilterTimeseries/'

def choose_ex_id(key='purpose', value='N2_aging'):
    ei = Experiment_Attribute_Index()
    return ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value)

def plot_filter_timeseries(blob_id, save=True, **kwargs):
    savename = ''
    if save:
        savename = '{d}bid{bid}.png'.format(d=SAVE_DIR, bid=blob_id)
        print savename

    width_document = pull_data_type_for_blob(blob_id, 'width50', **kwargs)
    width_timedict = width_document['data']
    length_timedict = pull_data_type_for_blob(blob_id, 'length', **kwargs)['data']

    times, widths = timedict_to_list(width_timedict)
    _, lengths = timedict_to_list(length_timedict)

    plot_timecourse(times, widths, lengths, savename=savename)

def fit_gaussian(x, num_bins=200):
    print len(x)
    n, bins = np.histogram(x, num_bins, normed=1)
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    mu, sigma = flags_and_breaks.fit_gaussian(x)
    y = mlab.normpdf(bincenters, mu, sigma)
    return bincenters, y

def plot_timecourse(times, widths, lengths, num_bins=30, savename=''):

    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.8
    left_h = left + width + 0.02

    width_color = 'green'
    length_color = 'blue'
    bounds_color = 'black'
    flag_color = 'red'

    plt.figure(1)
    timeseries = plt.axes([left, bottom, width, height])
    timeseries.plot(times, widths, color=width_color)
    timeseries.plot(times, lengths, color=length_color)
    timeseries.set_xlim([times[0], times[-1]])

    # Set up histograms
    axHisty = plt.axes([left_h, bottom, 0.2, height])
    axHisty.yaxis.tick_right()

    # Make width histogram
    ny, binsy = np.histogram(widths, num_bins, range=[np.min(widths), np.max(widths)], normed=True)
    bincenters = 0.5 * (binsy[1:] + binsy[:-1])
    new_bincenters = [binsy[0]]
    new_ny = [0]
    for N, bc in izip(ny, bincenters):
        new_bincenters.append(bc)
        new_ny.append(N)
    new_bincenters.append(binsy[-1])
    new_ny.append(0)
    print new_ny
    axHisty.fill_between(new_ny, new_bincenters, timeseries.get_ylim()[-1], facecolor=width_color, alpha=0.2)
    axHisty.plot(ny, bincenters, color=width_color, alpha=0.2)

    min_threshold, max_threshold = flags_and_breaks.calculate_threshold(widths, p=0.05)
    timeseries.plot([times[0], times[-1]], [min_threshold, min_threshold], color=bounds_color, ls='--')
    timeseries.plot([times[0], times[-1]], [max_threshold, max_threshold], color=bounds_color, ls='--')

    top_flags = flags_and_breaks.flag_outliers_in_timeseries(times, widths, options='long')
    top_flags = [float(i) for i in top_flags if not top_flags[i]]
    y = [max_threshold for _ in top_flags]
    timeseries.plot(top_flags, y, lw=0, color=flag_color, marker='.')
    bottom_flags = flags_and_breaks.flag_outliers_in_timeseries(times, widths, options='short')
    bottom_flags = [float(i) for i in bottom_flags if not bottom_flags[i]]
    y = [min_threshold for _ in bottom_flags]
    timeseries.plot(bottom_flags, y, lw=0, color=flag_color, marker='.')

    fitbin, fity = fit_gaussian(widths)
    axHisty.plot(fity, fitbin, color=bounds_color)

    # make length histogram
    ny, binsy = np.histogram(lengths, num_bins, range=[np.min(lengths), np.max(lengths)], normed=True)
    bincenters = 0.5 * (binsy[1:] + binsy[:-1])
    new_bincenters = [binsy[0]]
    new_ny = [0]
    for N, bc in izip(ny, bincenters):
        new_bincenters.append(bc)
        new_ny.append(N)
    new_bincenters.append(binsy[-1])
    new_ny.append(0)
    axHisty.fill_between(new_ny, new_bincenters, timeseries.get_ylim()[-1], facecolor=length_color, alpha=0.2)
    axHisty.plot(ny, bincenters, color=length_color, alpha=0.2)

    min_threshold, max_threshold = flags_and_breaks.calculate_threshold(lengths, p=0.05)
    timeseries.plot([times[0], times[-1]], [min_threshold, min_threshold], color=bounds_color, ls='--')
    timeseries.plot([times[0], times[-1]], [max_threshold, max_threshold], color=bounds_color, ls='--')

    top_flags = flags_and_breaks.flag_outliers_in_timeseries(times, lengths, options='long')
    top_flags = [float(i) for i in top_flags if not top_flags[i]]
    y = [max_threshold for _ in top_flags]
    timeseries.plot(top_flags, y, lw=0, color=flag_color, marker='.')
    bottom_flags = flags_and_breaks.flag_outliers_in_timeseries(times, lengths, options='short')
    bottom_flags = [float(i) for i in bottom_flags if not bottom_flags[i]]
    y = [min_threshold for _ in bottom_flags]
    timeseries.plot(bottom_flags, y, lw=0, color=flag_color, marker='.')

    fitbin, fity = fit_gaussian(lengths)
    axHisty.plot(fity, fitbin, color=bounds_color)

    timeseries.set_xlabel('time (s)')
    timeseries.set_ylabel('pixels')
    axHisty.set_xlabel('frequency')

    xmin, xmax = axHisty.get_xlim()
    axHisty.set_xlim([0, xmax])

    if savename:
        plt.savefig(savename)
    else:
        plt.show()
    plt.clf()


def main():
    mongo_client, _ = mongo.start_mongo_client(settings.MONGO['mongo_ip'], settings.MONGO['mongo_port'],
                                               settings.MONGO['worm_db'], settings.MONGO['blob_collection'])
    try:
        ex_ids = choose_ex_id()
        for ex_id in ex_ids[5:6]:
            print ex_id
            blobs = [e['blob_id'] for e in mongo_query({'ex_id': ex_id, 'data_type': 'smoothed_spine'}, {'blob_id':1})]
            for blob_id in blobs:
                print blob_id
                plot_filter_timeseries(blob_id, mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)


def plot_one_blob():
    mongo_client, _ = mongo.start_mongo_client(settings.MONGO['mongo_ip'], settings.MONGO['mongo_port'],
                                               settings.MONGO['worm_db'], settings.MONGO['blob_collection'])

    try:
        blob_id = '20130318_153742_02796'
        #blob_id = '20130320_102312_10828'
        #blob_id = '20130320_102312_19189'

        plot_filter_timeseries(blob_id, save=False, mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)

if __name__ == '__main__':
    plot_one_blob()
    #main()
