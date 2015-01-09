#!/usr/bin/env python
'''
Filename: file_manager.py

Description: holds many low-level scripts for finding, sorting, and saving files
in a rigid directory structure.
'''
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard library
import os
import sys
import json
from glob import iglob
import datetime
import errno

# third party
import pandas as pd
import numpy as np

# nonstandard imports
from waldo.conf import settings
from waldo.annotation.experiment_index import Experiment_Attribute_Index, organize_plate_metadata

from .prepdata import PrepData
from . import paths

DSET_OPTIONS = ['d', 'ds', 'dset', 'dataset', 's', 'data_set']
RECORDING_OPTIONS = ['p', 'plate', 'ex_id', 'eid']
WORM_OPTIONS = ['w', 'worm', 'blob', 'b', 'bid', 'blob_id']

class ImageMarkings(object):
    # a class for interacting with preprocessing data
    # for an recording (ie. experiment id or ex_id)
    # or for a dataset (ie. dataset name or dset)

    # this gives access to region of interest data
    # (x, y, and radius) and threshold data

    def __init__(self, ex_id):
        # specifiy the experiment.
        if not ex_id:
            raise ValueError('user must specify ex_id')

        self.ex_id = ex_id
        self.file = paths.threshold_data(ex_id)

        try:
            with self.file.open() as f:
                d = json.load(f)
        except:
            d = {}
        self.data = {'threshold': d.get('threshold', 0),
                     'x': d.get('x', 0),
                     'y': d.get('y', 0),
                     'r': d.get('r', 1)}

    def dump(self, data):
        with self.file.open('w') as f:
            json.dump(f, data)

    def roi(self, ex_id=None):
        return {'x': self.data['x'], 'y': self.data['y'],
                'r': self.data['r']}

    def threshold(self, ex_id=None):
        return self.data['threshold']

def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def df_equal( df1, df2 ):
    """ Check if two DataFrames are equal, ignoring nans """
    return df1.fillna(1).sort(axis=1).eq(df2.fillna(1).sort(axis=1)).all().all()

ensure_dir_exists = paths.mkdirp # alias
