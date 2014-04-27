#!/usr/bin/env python

'''
Filename: datasets.py
Description:

These functions are for consolidating plate summaries,
The plate summaries were created using plates.py
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
from itertools import izip
import numpy as np
import scipy.stats as stats
import pandas as pd

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.file_manager import get_plate_files #, read_plate_timeseries
from wio.file_manager import return_flattened_plate_timeseries, write_dset_summary
# format_dset_summary_name
from wio.file_manager import format_dirctory, ensure_dir_exists, get_dset
from wio.file_manager import write_table, read_table
from metrics.measurement_switchboard import STANDARD_MEASUREMENTS
from annotation.experiment_index import Experiment_Attribute_Index2
from wio.file_manager import get_annotations

# globals
XLIMS = {'cent_speed_bl': [0.0, 0.04],
         'length_mm': [0.0, 1.5],
         'curve_bl': [0.0, 0.006],
         'curve_w': [0.0, 0.04],
         'width_mm': [0.0, .2],
         'angle_change': [-0.1, 0.1],
         'stop_dur': [0, 10],
         'go_dur': [0, 100]}


# def generate_distribution(dataset, data_type, label, xlim, verbose=True):
#     ex_ids, days, dfiles = get_annotations(dataset=dataset, data_type=data_type, label=label)
#     print '{l}: {N} recordings found'.format(l=label, N=len(ex_ids))
#     #organize data by days
#     #data_by_days = organize_plates_by_day(ex_ids, dfiles, days)
#
#     data_by_days = {}
#     for e, d, f in zip(ex_ids, days, dfiles):
#         day = int(d[1:])
#         if day not in data_by_days:
#             data_by_days[day] = []
#         data_by_days[day].append((e, f))
#
#
#
#     # get a distribution for each day.
#     day_distributions = {}
#     day_quartiles = {}
#     for day in sorted(data_by_days)[:]:
#
#         all_data = []
#         for eID, f in data_by_days[day][:]:
#             plate_data = return_flattened_plate_timeseries(eID, dataset, data_type)
#             all_data.extend(list(plate_data))
#
#
#         if verbose:
#             #print 'day', day, 'recordings:', len(data_by_days[day]), 'timepoints:', len(all_data)
#             print '\tday {d} | recordings:{r} | timepoints: {t}'.format(d=day,
#                                                                         r=len(data_by_days[day]),
#                                                                         t=len(all_data))
#         s = all_data
#         xmin, xmax = xlim
#         bins = np.linspace(xmin, xmax, 5000)
#         y, x = np.histogram(s, bins=bins)
#         x = x[1:]
#         y = np.array(y, dtype=float) / sum(y)
#
#         day_distributions[day] = {'x': list(x), 'y': list(y)}
#         #day_quartiles[day] = [stats.scoreatpercentile(all_data, 25),
#         #                      stats.scoreatpercentile(all_data, 50),
#         #                      stats.scoreatpercentile(all_data, 75)]
#     print 'writing'
#     write_dset_summary(data=day_distributions, sum_type='dist', ID=label,
#                        data_type=data_type, dataset=dataset)


def generate_distribution(dataset, data_type, label, xlim, verbose=True):
    ex_ids, days, dfiles = get_annotations(dataset=dataset, data_type=data_type, label=label)
    print '{l}: {N} recordings found'.format(l=label, N=len(ex_ids))
    #organize data by days
    data_by_days = {}
    for e, d in izip(ex_ids, days):
        day = int(d[1:])
        if day not in data_by_days:
            data_by_days[day] = []
        data_by_days[day].append(e)

    # get a distribution for each day.

    # initialize data-frames
    xmin, xmax = xlim
    bins = np.linspace(xmin, xmax, 5000)
    day_distributions = pd.DataFrame(index=bins)
    day_quartiles = pd.DataFrame(index=['q1', 'm', 'q3'])
    # calculate distributions across all days and add to dataframes
    for day in sorted(data_by_days):
        all_data = []
        for eID in data_by_days[day][:]:
            plate_data = return_flattened_plate_timeseries(eID, dataset, data_type)
            all_data.extend(list(plate_data))

        if verbose:
            print '\tday {d} | recordings:{r} | timepoints: {t}'.format(d=day,
                                                                        r=len(data_by_days[day]),
                                                                        t=len(all_data))

        Ns, bins = np.histogram(all_data, bins=bins)
        day_distributions[day] = np.array(Ns, dtype=float) / sum(Ns)
        day_quartiles[day] = [stats.scoreatpercentile(all_data, 25),
                              stats.scoreatpercentile(all_data, 50),
                              stats.scoreatpercentile(all_data, 75)]

    print 'writing'
    #write_dset_summary(data=day_distributions, sum_type='dist', ID=label,
    #                   data_type=data_type, dataset=dataset

    write_table(ID=label, ID_type='dset', dataframe=day_distributions, data_type=data_type,
                dset=dataset, file_tag='dist')

    write_table(ID=label, ID_type='dset', dataframe=day_quartiles, data_type=data_type,
                dset=dataset, file_tag='quartiles')

def preprocess_distribution_set(dataset, labels=None,
                                data_types=STANDARD_MEASUREMENTS):
    if labels == None:
        ei = Experiment_Attribute_Index2(dataset)
        labels = [str(i) for i in set(ei['label'])]
        labels.append('all')

    print 'preprocessing distributions'
    print 'labels: {ls}'.format(ls=', '.join(labels))
    for data_type in data_types:
        xlim = XLIMS.get(data_type, [0, 1])
        for label in labels:
            generate_distribution(dataset, data_type, label, xlim=xlim)


# def combine_worm_percentiles_for_dset2(dataset):
#     """
#
#     Note: currently takes first percentiles file to be current
#     and expects all rest to have same format.
#     """
#     data_type = 'percentiles'
#     tag = 'worm_percentiles'
#     ex_ids, plate_files = get_plate_files(dataset=dataset,
#                                           data_type=data_type,
#                                           tag=tag)
#
#     all_blob_ids, all_percentiles = [], None
#     expected_N_cols = 0
#     for ex_id in ex_ids:
#         #ex_id = pf.split('/')[-1].split('-')[0]
#         #print ex_id
#         blob_ids, percentiles = read_plate_timeseries(ex_id,
#                                                       dataset=dataset,
#                                                       data_type=data_type,
#                                                       tag=tag)
#         N_rows, N_cols = percentiles.shape
#         if N_rows ==0 or N_cols==0:
#             continue
#         if all_percentiles == None:
#             all_blob_ids = list(blob_ids)
#             all_percentiles = percentiles
#             expected_N_cols = N_cols
#         elif N_cols == expected_N_cols:
#             all_blob_ids.extend(list(blob_ids))
#             all_percentiles = np.concatenate((all_percentiles, percentiles))
#         else:
#             print 'warning: columns mismatch between'
#             print (ex_ids[0], expected_N_cols), 'and', (ex_id, N_cols)
#
#         #print len(all_blob_ids), all_percentiles.shape
#
#     nan_count = 0
#     data_types = []
#     for i, row in enumerate(all_percentiles):
#         for i in row:
#             data_types.append(type(i))
#             if np.isnan(i):
#                 nan_count += 1
#                 print 'nan in row:', i
#
#     print 'total NaNs found:', nan_count
#     print 'types found:', list(set(data_types))
#     return all_blob_ids, all_percentiles


# def consolidate_dset_from_plate_timeseries(dataset, data_type, verbose=True):
#     ex_ids, dfiles = get_plate_files(dataset, data_type)
#     means, stds = [], []
#     quartiles = []
#     hours = []
#     days = []
#     #labels, sublabels, plate_ids = {}, {}, {}
#     labels, sublabels, plate_ids = [], [], []
#
#     for i, (ex_id, dfile) in enumerate(izip(ex_ids, dfiles)):
#         hour, label, sub_label, pID, day = organize_plate_metadata(ex_id)
#         hours.append(hour)
#         labels.append(label)
#         sublabels.append(sub_label)
#         plate_ids.append(pID)
#         days.append(day)
#
#         flat_data = return_flattened_plate_timeseries(ex_id, dataset, data_type)
#         if not len(flat_data):
#             continue
#         means.append(np.mean(flat_data))
#         stds.append(np.std(flat_data))
#         men = float(np.mean(flat_data))
#         #print men, type(men), men<0
#         #print flat_data[:5]
#         quartiles.append([stats.scoreatpercentile(flat_data, 25),
#                           stats.scoreatpercentile(flat_data, 50),
#                           stats.scoreatpercentile(flat_data, 75)])
#         if verbose:
#             print '{i} | {eID} | N: {N} | hour: {h} | label: {l}'.format(i=i, eID=ex_id,
#                                                                          N=len(flat_data),
#                                                                          h=round(hour, ndigits=1),
#                                                                          l=label)
#
#     #for i in zip(ex_ids, means, stds, quartiles):
#     #    print i
#     data={'ex_ids':ex_ids,
#           'hours':hours,
#           'mean':means,
#           'std':stds,
#           'quartiles':quartiles,
#           'labels':labels,
#           'sub':sublabels,
#           'plate_ids':plate_ids,
#           'days':days,
#           }
#     return data

def consolidate_dset_from_plate_timeseries(dataset, data_type, verbose=True):
    """

    :param dataset:
    :param data_type:
    :param verbose:
    :return:
    """
    ex_ids, dfiles = get_plate_files(dataset, data_type)

    data = {'ex_ids': [], 'hours': [], 'labels': [],
            'sub': [], 'plate_ids': [], 'days': [],
            'mean': [], 'std': [],
            'q1': [], 'median': [], 'q3': []}

    for i, ex_id in enumerate(ex_ids):

        flat_data = return_flattened_plate_timeseries(ex_id, dataset, data_type)
        if not len(flat_data):
            continue

        hour, label, sub_label, pID, day = organize_plate_metadata(ex_id)

        data['ex_ids'].append(ex_id)
        data['hours'].append(hour)
        data['labels'].append(label)
        data['sub'].append(sub_label)
        data['plate_ids'].append(pID)
        data['days'].append(day)

        data['mean'].append(np.mean(flat_data))
        data['std'].append(np.std(flat_data))
        data['q1'].append(stats.scoreatpercentile(flat_data, 25))
        data['median'].append(stats.scoreatpercentile(flat_data, 50))
        data['q3'].append(stats.scoreatpercentile(flat_data, 75))

        if verbose:
            print '{i} | {eID} | N: {N} | hour: {h} | label: {l}'.format(i=i, eID=ex_id,
                                                                         N=len(flat_data),
                                                                         h=round(hour, ndigits=1),
                                                                         l=label)
    return pd.DataFrame(data, index_col=['ex_ids'])

# MAIN FUNCTION
def write_combined_worm_percentiles(dataset):
    # manage paths for files
    save_dir = format_dirctory(ID=dataset, ID_type='dset')
    f_savename = '{path}{dset}_features.csv'.format(path=save_dir, dset=dataset)
    i_savename = '{path}{dset}_index.csv'.format(path=save_dir, dset=dataset)
    ensure_dir_exists(save_dir)
    # combine worm percentiles and indexes for worms

    # Note: currently takes first percentiles file to be current
    # and expects all rest to have same format.
    ex_ids, plate_files = get_plate_files(dataset=dataset, data_type='percentiles',
                                          tag='worm_percentiles')
    get_plate = lambda x: read_table(ID=x, ID_type='plate', data_type='percentiles',
                                     dset=dataset, file_tag='worm_percentiles')
    plate_percentiles = [get_plate(ex_id) for ex_id in ex_ids
                         if get_plate(ex_id) is not None]
    if not len(plate_percentiles):
        print 'no data found for {dset}'.format(dset=dataset)
        return
    percentiles = pd.concat(plate_percentiles)
    print percentiles.columns
    blob_ids = percentiles.index

    # write features
    percentiles.to_csv(f_savename, header=False)

    # write worm index
    ex_ids = ['_'.join(bi.split('_')[:2]) for bi in blob_ids]
    ex_id_data = {}
    for ex_id in list(set(ex_ids)):
        ex_id_data[ex_id] = organize_plate_metadata(ex_id)
    worm_index = [ex_id_data[ex_id] for ex_id in ex_ids]
    print '{N} blob ids included'.format(N=len(blob_ids))

    wi = pd.DataFrame(worm_index, index=blob_ids)
    wi.to_csv(i_savename, header=False)

    for line in worm_index[:5]:
        print line
    print '...'

# def write_combined_worm_percentiles2(dataset):
#     # manage paths for files
#     save_dir = format_dirctory(ID=dataset, ID_type='dset')
#     f_savename = '{path}{dset}_features.csv'.format(path=save_dir, dset=dataset)
#     i_savename = '{path}{dset}_index.csv'.format(path=save_dir, dset=dataset)
#     ensure_dir_exists(save_dir)
#     # combine worm percentiles and indexes for worms
#     blob_ids, percentiles = combine_worm_percentiles_for_dset(dataset)
#     #worm_index = create_full_worm_index(blob_ids)
#     ex_id_data, worm_index = {}, []
#     for blob_id in blob_ids:
#         ex_id = '_'.join(blob_id.split('_')[:2])
#         #print ex_id
#         if ex_id not in ex_id_data:
#             ex_id_data[ex_id] = organize_plate_metadata(ex_id)
#         worm_index.append(ex_id_data[ex_id])
#     print '{N} blob ids included'.format(N=len(blob_ids))
#     # write features
#     pf = pd.DataFrame(percentiles, index=blob_ids)
#     pf.to_csv(f_savename, header=False)
#     # write worm index
#     wi = pd.DataFrame(worm_index, index=blob_ids)
#     wi.to_csv(i_savename, header=False)
#
#     for line in worm_index:
#         print line

# MAIN FUNCTION
def write_dset_summaries(dataset, data_types=STANDARD_MEASUREMENTS):
    for data_type in data_types:
        print 'consolidating data for {dt}'.format(dt=data_type)
        data = consolidate_dset_from_plate_timeseries(dataset, data_type)
        print dataset, data_type, data
        summary = pd.DataFrame(data)
        print summary.head()
        #write_dset_summary(data=data, sum_type='basic',
        #                   data_type=data_type, dataset=dataset)

        write_table(ID=dataset,
                    ID_type='dset',
                    dataframe=df,
                    data_type=data_type,
                    dset=dataset,
                    file_tag='basic')

        # # from read table
        # filename = format_filename(ID=ID,
        #                            ID_type=ID_type,
        #                            data_type=data_type,
        #                            file_type=file_type,
        #                            dset=dset,
        #                            file_tag=file_tag,
        #                            file_dir=file_dir)
        # filename = format_filename(ID=ID, ID_type='dset',
        #                            data_type=data_type,
        #                            file_tag=sum_type,
        #                            dset=dataset,
        #                            file_dir=dset_dir,
        #                            file_type='json')


if __name__ == '__main__':
    dataset = 'disease_models'
    dataset = 'N2_aging'
    #dataset = 'thermo_recovery'
    #dataset = 'copas_TJ3001_lifespan'
    #write_combined_worm_percentiles(dataset)
    #write_dset_summaries(dataset)
    #generate_distribution(dataset)
    preprocess_distribution_set(dataset)
