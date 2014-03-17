#!/usr/bin/env python
'''
Filename: process_plate_timeseries.py
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
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
#from exponential_fitting import fit_exponential_decay_robustly, rebin_data, exponential_decay, fit_constrained_decay_in_range
#from plate_utilities import get_ex_id_files,  write_dset_summary, parse_plate_timeseries_txt_file
#from plate_utilities import return_flattened_plate_timeseries, organize_plate_metadata
from wio.file_manager import format_results_filename


if __name__ == '__main__':
    ID = 'id'
    result_type='multi_path'
    tag= None#'size'
    dset='dataset1'
    ID_type='dset'
    print format_results_filename(ID, result_type, tag,
                                  dset, ID_type)
    
