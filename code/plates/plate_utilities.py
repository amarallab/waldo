#!/usr/bin/env python

'''
Filename: plate_utilities.py
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

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from settings.local import LOGISTICS
from wio.file_manager import ensure_dir_exists

# Globals
PLATE_DIR = os.path.abspath(LOGISTICS['data'] + 'plate_summary')
DSET_DIR = os.path.abspath(LOGISTICS['data'] + 'dsets')
TIME_SERIES_DIR = os.path.abspath(LOGISTICS['export'])

ensure_dir_exists(PLATE_DIR)
ensure_dir_exists(TIME_SERIES_DIR)

def show_timeseries_options(timeseries_dir=TIME_SERIES_DIR):
    print '\ndata_set\tdata_type\n'
    data_dirs = glob.glob('{path}/*'.format(path=timeseries_dir.rstrip('/')))
    for d in sorted(data_dirs):
        if os.path.isdir:
            d = d.split('/')[-1]
            data_set, data_type = d.split('-')
            print data_set,'\t',  data_type

def get_ex_id_files(dataset, data_type, path=TIME_SERIES_DIR):
    search = '{path}/{ds}-{dt}/*'.format(path=path.rstrip('/'),
                                         ds=dataset, dt=data_type)
    ex_ids, file_paths = [], []
    for file_path in glob.glob(search):
        if os.path.isfile(file_path):
            ex_id = file_path.split('/')[-1].split('-' +data_type)[0]
            # one small check before storing
            if len(ex_id.split('_')) == 2:
                ex_ids.append(ex_id)
                file_paths.append(file_path)
    return ex_ids, file_paths
    
def format_dset_summary_name(data_type, dataset, sum_type, dset_dir):
    ensure_dir_exists(dset_dir)
    path = '{setdir}/{dset}-{dtype}-{stype}.json'.format(setdir=dset_dir,
                                                         dset=dataset,
                                                         dtype=data_type,
                                                         stype=sum_type)
    print path
    return path


def write_dset_summary(data, data_type, dataset, sum_type, dset_dir=DSET_DIR):
    filename = format_dset_summary_name(data_type, dataset, sum_type, dset_dir)
    json.dump(data, open(filename, 'w'))

def read_dset_summary(data_type, dataset, sum_type='basic', dset_dir=DSET_DIR):
    filename = format_dset_summary_name(data_type, dataset, sum_type, dset_dir)
    return json.load(open(filename, 'r'))


def format_plate_summary_name(ex_id, sum_type, dataset, data_type, path):
    savedir = '{path}/{dset}-{dtype}'.format(path=path, dset=dataset, 
                                             dtype=data_type)
    ensure_dir_exists(savedir)
    filename = '{savedir}/{eID}.json'.format(savedir=savedir, eID=ex_id)                                             
    return filename

def write_plate_summary(data, ex_id, sum_type, dataset, data_type, path=PLATE_DIR):
    filename = format_plate_summary_name(ex_id, sum_type, dataset, data_type, path)
    json.dump(data, open(filename, 'w'))

def read_plate_summaryo(sum_type, dataset, data_type, path=PLATE_DIR):
    filename = format_plate_summary_name(ex_id, sum_type, dataset, data_type, path)
    if os.path.isfile(filename):
        return json.load(open(filename, 'r'))
    return None

if __name__ == '__main__':
    dataset = 'disease_models'
    data_type = 'cent_speed_bl'
    #show_timeseries_options()
    ex_ids, file_paths = get_ex_id_files(dataset, data_type)
