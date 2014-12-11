#!/usr/bin/env python
'''
Filename:
Description:


'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import glob
import pandas as pd

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

from annotation.experiment_index import Experiment_Attribute_Index2
from conf import settings

def count_file_types(basedir):
    counts = {}
    for f in glob.iglob(basedir + '*'):
        bID, filetype = os.path.basename(f).split('-')
        if filetype not in counts:
            counts[filetype] = 0
        counts[filetype] += 1
    return counts

def merge_experiment_index_with_file_counts(dataset, data_dir=None):
    if data_dir is None:
        data_dir = settings.LOGISTICS['data']
    data_dir = os.path.abspath(data_dir)
    ei = Experiment_Attribute_Index2(dataset=dataset)

    data, null_recordings = {}, []
    for eID in list(ei.index):
        search_dir = '{d}/worms/{eId}/'.format(d=data_dir, eId=eID)
        counts = count_file_types(search_dir)
        data[eID] = counts
        if counts == {}:
            null_recordings.append(eID)

    print 'found {N} recordings with no data'.format(N=len(null_recordings))
    if len(null_recordings) == len(ei.index):
        return []
    #print null_recordings
    results = pd.DataFrame(data).T
    return pd.concat([ei, results], axis=1)

def show_counts_by_label(results, final_filetype, verbose=True):
    labels = list(set(results['label']))
    ages = list(set(results['age']))
    r = results
    summary = pd.DataFrame(index=['label','age', 'recordings', 'worms'])
    i = 0
    row_index = []
    for l in sorted(labels):
        for a in sorted(ages):
            sub_section = r[r['age'] == a][r['label'] ==l]
            a = int(a[1:])
            col_name = '{l}-{a}'.format(l=l, a=a)
            N = len(sub_section)
            N_worms = sub_section[final_filetype].sum()
            summary[col_name] = [l, a, N, N_worms]
            row_index.append((l, a))
            i += 1

    summary = summary.T.sort(['label', 'age']).reset_index(drop=True)
    if verbose:
        print summary
    return summary

def file_counts_dataframe(dataset, data_dir=None):
    if data_dir is None:
        data_dir = settings.LOGISTICS['data']
    data_dir = os.path.abspath(data_dir)
    print data_dir
    ei = Experiment_Attribute_Index2(dataset=dataset)
    data = {}
    null_recordings = []
    for eID in list(ei.index):
        search_dir = '{d}/worms/{eId}/'.format(d=data_dir, eId=eID)
        counts = count_file_types(search_dir)
        data[eID] = counts
        if counts == {}:
            null_recordings.append(eID)

    print 'found {N} recordings with no data'.format(N=len(null_recordings))
    if len(null_recordings) == len(ei.index):
        return []
    #print null_recordings
    return pd.DataFrame(data).T

def show_completeness(results):

    cols = [col for col in ['metadata.json', 'spine_rough.h5', 'spine.h5']
            if col in results.columns]
    #print results[['metadata.json', 'spine_rough.h5', 'spine.h5']]
    print results[cols]
    summary = pd.DataFrame()
    summary['fraction'] = results.sum() / results['metadata.json'].sum()
    summary['missing'] = results['metadata.json'].sum() - results.sum()
    return summary

def show_dset(dataset):
    results = merge_experiment_index_with_file_counts(dataset)
    if len(results) == 0:
        return
    fin_file = 'spine.h5'
    if fin_file not in results.columns:
        fin_file = 'xy.h5'
    show_counts_by_label(results, final_filetype=fin_file)

def show_dset_completeness(dataset):
    results = file_counts_dataframe(dataset)
    if len(results) > 0:
        print show_completeness(results)


if __name__ == '__main__':
    # toggles
    dataset = 'disease_models'
    #dataset = 'N2_aging'
    dataset = 'thermo_recovery'

    show_dset(dataset)
    show_dset_completeness(dataset)
