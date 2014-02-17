#!/usr/bin/env python

'''
Filename: temp_cache.py

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
PROJECT_HOME = os.path.abspath(HERE + '/../../')
CODE_DIR = os.path.abspath(HERE + '/../')
SHARED_DIR = CODE_DIR + 'shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports

from wormmetrics.switchboard import switchboard
from database.mongo_retrieve import pull_data_type_for_blob, timedict_to_list

def pull_blob_data(blob_id, metric):
    ''' returns a list of times and a list of data for a given blob_id and metric.

    This function chooses which program to call in order to calculate or retrieve
    the desired metric.
    '''
    pull_data = switchboard(metric=metric, harsh=False)
    if pull_data:
        data_timedict = pull_data(blob_id, metric=metric, for_plotting=True)
    else:
        data_timedict = pull_data_type_for_blob(blob_id, data_type=metric, **kwargs)['data']
    times, data = timedict_to_list(data_timedict)
    return times, data
