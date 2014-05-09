# -*- coding: utf-8 -*-
"""
Gets data from processed HDF5 files based on the blob ID
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import sys
import os.path as op

PROJECT_DIR = op.abspath(op.join(op.dirname(op.realpath(__file__)), '..', '..', '..', '..'))
CODE_DIR = op.join(PROJECT_DIR, 'code')
SHARED_DIR = op.join(PROJECT_DIR, 'code', 'shared')
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

WALDO_LOC = op.join(op.dirname(__file__), '..', 'Waldo')
WALDO_CODE = op.join(WALDO_LOC, 'code')
WALDO_DATA = op.join(WALDO_LOC, 'data', 'worms')

import wio.blob_reader

from .adapter import WormDataAdapter
from .util import harmonize_id

# def waldo(data_set, bid):
#     sys.path.append(WALDO_CODE)
#     from shared.wio.file_manager import get_timeseries

#     ext_bid = '{}_{:05d}'.format(data_set, bid)

#     return get_timeseries(ext_bid, 'xy')


def iter_through_worms(ex_id, data_type, blob_ids=None):
    ''' iter through a series of blob_ids for a given ex_id and dataset,
    yeilds a tuple of (blob_id, times, data) of all blob_ids

    params
    ex_id: (str)
        id specifying which recording you are examining.
    data_type: (str)
        type of data you would like returned. examples: 'length_mm', 'spine', 'xy'
    blob_ids: (list of str or None)
       the blob_ids you would like to check for the datatype. by default all blobs with existing files are checked.
    '''
    if blob_ids == None:
        blob_ids = get_good_blobs(ex_id=ex_id, key=data_type)
    print('{N} blob_ids found'.format(N=len(blob_ids)))
    for blob_id in blob_ids:
        times, data = pull_blob_data(blob_id, metric=data_type)
        if times != None and len(times):
            #print blob_id
            yield blob_id, times, data


class WaldoAdapter(WormDataAdapter):
    def locate(self):
        'find worms'
