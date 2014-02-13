#!/usr/bin/env python

'''
Filename: plot_rows_of_distrubutions.py
Description: 
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import datetime
import glob
import json

# Path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_directory)

# nonstandard imports
from Shared.Code.Plotting.TraitAggregation.multi_histogram import table_histogram_set
from Shared.Code.Plotting.TraitAggregation.multi_histogram import sideways_hist_plot

DATA_DIR = project_directory + 'Shared/Data/Export/'
save_dir = './../Results/Distributions/'

def read_index(data_dir=DATA_DIR):
    index = json.load(open(data_dir + 'index.json', 'r'))
    return index


def read_data_from_files(data_type='centroid_speed', decile=4, index_key='age', index=None):


    if not index:
        index = read_index()

    data = {}
    for dfile in glob.glob(DATA_DIR + 'blob*.json'):
        print dfile.split('/')[-1]
        try:
            file_data = json.load(open(dfile, 'r'))
            for blob_id, blob_data in file_data.iteritems():
                index_bin = index[blob_id][index_key]
                if index_key == 'age':
                    index_bin = int(index_bin.lstrip('A'))
                data_point = blob_data[data_type][decile]
                if index_bin not in data:
                    data[index_bin] = []
                data[index_bin].append(data_point)
        except Exception as e:
            print dfile, e

    datasets, labels =[], []
    for dkey in sorted(data):
        datasets.append(data[dkey])
        labels.append(dkey)
    return datasets, labels


if __name__ == '__main__':
    #index = read_index()
    num_to_plot = 9
    data_types = ['smooth_length',
                  'curvature_all',
                  'speed_perp',
                  'speed_along',
                  'centroid_speed',
                  'width_mm',
                  'curvature_all_bl',
                  'speed_perp_bl',
                  'speed_along_bl',
                  'centroid_speed_bl',
                  'centroid_ang_ds',
                  'width_bl',
                  'size_mm2']

    for dt in data_types[:1]:
        title = 'median values of \'{dt}\' across age'.format(dt=dt.replace('_', ' '))
        save_name = '{save_dir}ageing_{dt}.png'.format(save_dir=save_dir,dt=dt)
        print title
        print save_name
        datasets, labels = read_data_from_files(data_type=dt, decile=4)
        if len(datasets) > 1:
            table_histogram_set(datasets=datasets[:num_to_plot], labels=labels[:num_to_plot], savename=save_name, title=title)
            sideways_hist_plot(datasets=datasets[:num_to_plot], labels=labels[:num_to_plot], savename='', title=title)


    # toggle to show 90th percentile for each worm.
    '''
    dt = 'centroid_speed'
    datasets, labels = read_data_from_files(data_type=dt, decile=8)
    table_histogram_set(datasets=datasets[:num_to_plot], labels=labels[:num_to_plot])
    #def table_histogram_set(datasets, labels, x_label=None, num_bins=50, savename=None, guidelines=True):
    '''
