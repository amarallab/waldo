#!/usr/bin/env python

'''
Filename: h5_interface.py

Description:
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

#standard imports
import os
import sys
import json
from glob import glob
from itertools import izip
import h5py

'''
# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)
'''

OPTIONS = {}

def write_h5_timeseries_base(h5_file, h5_path, times, data):
    pass

def read_h5_timeseries_base(h5_file, h5_path):
    times, data = [], []
    return times, data

def delete_h5_dataset(h5_file, h5_path, times, data):
    pass
