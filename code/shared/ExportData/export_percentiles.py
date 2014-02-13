#!/usr/bin/env python

'''
Filename: export_percentiles.py
Description: Functions to write a json for each ex_id containing every processed blob
with percentiles for every measurement.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import json
from scipy.stats import scoreatpercentile
import time

# path definitions
PROJECT_DIR =  os.path.dirname(os.path.realpath(__file__)) + '/../../../'
CODE_DIR = PROJECT_DIR + 'code/'
SHARED_DIR = CODE_DIR + 'shared/'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
import WormMetrics.switchboard as sb
import database.mongo_retrieve as mr
from settings.local import LOGISTICS

# Globals
OUTDIR = LOGISTICS['export_dir']

def export_blob_percentiles_by_ex_id(ex_id, out_dir=OUTDIR, verbose=True, **kwargs):
    """
    write a json for an ex_id in which each blob has a dictionary of measurements containing
    a list with the 10, 20, 30, ... 80, 90 th percentiles.

    :param ex_id: the ex_id to write a json for.
    :param out_dir: the directory to write to.
    """
    now_string = time.ctime().replace('  ', '_').replace(' ', '_').replace(':', '.').strip()
    save_name = '{path}/blob_percentiles_{eid}_{now}.json'.format(path=out_dir, now=now_string, eid=ex_id)
    blob_ids = mr.unique_blob_ids_for_query({'ex_id': ex_id, 'data_type': 'smoothed_spine'}, **kwargs)
    blob_data = {}
    N = len(blob_ids)
    for i, blob_id in enumerate(blob_ids):
        if verbose:
            print 'calculating metrics for {bID} ({i}/{N})'.format(bID=blob_id, i=i, N=N)
        blob_data[blob_id] = {}
        for metric, data in sb.pull_all_for_blob_id(blob_id, **kwargs).iteritems():
            #print len(data)
            blob_data[blob_id][metric] = [scoreatpercentile(data, i) for i in xrange(10, 100, 10)]
            json.dump(blob_data, open(save_name, 'w'), indent=4, sort_keys=True)
    json.dump(blob_data, open(save_name, 'w'), indent=4, sort_keys=True)

if __name__ == '__main__':
    ex_id = '00000000_000001'
    export_blob_percentiles_by_ex_id(ex_id)
