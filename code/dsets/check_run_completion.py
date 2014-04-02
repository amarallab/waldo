#!/usr/bin/env python
'''
Filename: compile_plates_for_clustering.py
Description:


'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import datetime
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import glob
import scipy.stats as stats
import pandas as pd

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

#from wio.plate_utilities import get_plate_files, read_plate_timeseries, organize_plate_metadata
#from wio.file_manager import format_dirctory, ensure_dir_exists
from annotation.experiment_index import Experiment_Attribute_Index2
from settings.local import LOGISTICS

def count_file_types(basedir):
    counts = {}
    for f in glob.iglob(basedir + '*'):
        bID, filetype = os.path.basename(f).split('-')
        if filetype not in counts:
            counts[filetype] = 0
        counts[filetype] += 1
    return counts

def file_counts_dataframe(dataset, data_dir):
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
    print null_recordings
    return pd.DataFrame(data).T

if __name__ == '__main__':
    # toggles
    dataset = 'disease_models'
    #dataset = 'N2_aging'
    data_dir = LOGISTICS['data']

    results = file_counts_dataframe(dataset, data_dir)
    #print results.columns
    print results[['metadata.json', 'spine_rough.h5', 'spine.h5']]

    summary = pd.DataFrame()
    summary['fraction'] = results.sum() / results['metadata.json'].sum()
    summary['missing'] = results['metadata.json'].sum() - results.sum()
    print summary
