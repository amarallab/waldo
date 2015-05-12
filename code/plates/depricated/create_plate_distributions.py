#!/usr/bin/env python

'''
Filename: fit_plate_timeseries.py
Description:
Pull one type of data out of database, and save it in jsons organized by ex_id.
data pulled is broken into 15 minute segments. within each 15min segment data is pulled either
by subsampling the data or by binning into 10 second bins.
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

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIRECTORY = os.path.abspath(HERE + '/../../')
sys.path.append(PROJECT_DIRECTORY)

# nonstandard imports
from exponential_fitting import fit_exponential_decay_robustly, rebin_data, exponential_decay, fit_constrained_decay_in_range
from Shared.Code.ExportData.export_utilities import ensure_dir_exists
from fit_plate_timeseries import read_plate_timeseries
TIME_SERIES_DIR = HERE + '/../Data/Time-Series/'
DIST_DIR = HERE + '/../Data/Distributions/'

def get_plate_distribution(data):
    plate_distribution = []
    for i, t_bin in enumerate(data):
        plate_distribution += list(t_bin)
    return plate_distribution

if __name__ == '__main__':
    #data_ID = 'N2_aging-curvature_all_bl'
    data_ID = 'N2_aging-centroid_speed'
    #data_ID = 'N2_aging-smooth_length'
    timeseries_dir = TIME_SERIES_DIR + data_ID + '/'
    save_name = DIST_DIR + data_ID + '.json'
    compiled_data = {}
    for i, (times, data, ex_id) in enumerate(read_plate_timeseries(timeseries_dir)):
        compiled_data[ex_id] = get_plate_distribution(data)
    else:
        # only save over data if loop finishes without a break.
        json.dump(compiled_data, open(save_name, 'w'), indent=4, sort_keys=True)
