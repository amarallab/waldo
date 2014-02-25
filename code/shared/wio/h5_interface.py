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
#from itertools import izip
 
# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from settings.local import LOGISTICS
from file_manager import ensure_dir_exists

INDEX_DIR = LOGISTICS['annotation']
EXPORT_PATH = LOGISTICS['export']
H5_DIR = PROJECT_HOME + '/data/h5-binaries/'


def format_h5_path(blob_id, data_type, h5dir=H5_DIR):
    file_path = 
    errmsg = 'blob_id must be string, not {i}'.format(i=blob_id)
    assert isinstance(blob_id, basestring), errmsg
    ex_id = '_'.join(blob_id.split('_')[:2])
    blob_path = '{path}/{eID}'.format(path=tmp_dir, eID=ex_id)
    ensure_dir_exists(blob_path)
    tmp_file = '{path}/{bID}-{dt}.json'.format(path=blob_path, bID=blob_id,
                                               dt=data_type)
def write_timeseries(blob_id, data_type, times, array, h5dir=H5_DIR):
    assert isinstance(array, np.array)
    
