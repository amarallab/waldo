#!/usr/bin/env python
# coding: utf-8
"""
Look at data from a plate of worms
"""

import sys
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.abspath('.')
CODES = os.path.abspath(os.path.join(HERE, '..'))
SHARED = os.path.join(CODES, 'shared')
print SHARED
sys.path.append(SHARED)
sys.path.append(CODES)

from conf import settings
from wio.file_manager import read_table, ensure_dir_exists, get_good_blobs, get_timeseries, get_dset
from metrics.measurement_switchboard import pull_blob_data

# TODO: eliminate redundency in function.
def pull_plate_from_phoenix(ex_id, pull=True):
    ''' pulls all plate-level and worm-level data from phoenix to local version of waldo.'''
    dataset = get_dset(ex_id)
    p_address = 'phoenix.research.northwestern.edu'
    # source
    p1 = '/home/projects/worm_movement/waldo/data/plates/{dset}/timeseries/{eid}*'.format(eid=ex_id, dset=dataset)
    p2 = '/home/projects/worm_movement/waldo/data/plates/{dset}/percentiles/{eid}*'.format(eid=ex_id, dset=dataset)
    w1 = '/home/projects/worm_movement/waldo/data/worms/{eid}'.format(eid=ex_id)
    # destination
    d_p1 = '{p_dir}timeseries'.format(p_dir=settings.LOGISTICS['plates'])
    d_p2 = '{p_dir}percentiles'.format(p_dir=settings.LOGISTICS['plates'])
    d_w1 = '{w_dir}'.format(w_dir=settings.LOGISTICS['worms'])
    # make sure destination exists
    for d in [d_p1, d_p2, d_w1]:
        print d
        ensure_dir_exists(d)
    # commands
    cmd_p1 = 'scp -v {address}:{cmd} {dest}'.format(address=p_address, cmd=p1, dest=d_p1)
    cmd_p2 = 'scp -v {address}:{cmd} {dest}'.format(address=p_address, cmd=p2,  dest=d_p2)
    cmd_w1 = 'scp -rv {address}:{cmd} {dest}'.format(address=p_address, cmd=w1,  dest=d_w1)
    # run all the commands
    for cmd in [cmd_p1, cmd_p2, cmd_w1]:
        print
        print cmd
        print
        if pull:
            os.system(cmd)

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
    print '{N} blob_ids found'.format(N=len(blob_ids))
    for blob_id in blob_ids:
        times, data = pull_blob_data(blob_id, metric=data_type)
        if times != None and len(times):
            #print blob_id
            yield blob_id, times, data

def plot_all(ex_id, data_type):
    ''' example script for pulling and plotting worm data '''
    fig, ax = plt.subplots()
    for bid, times, data in iter_through_worms(ex_id, data_type):
        print bid
        ax.plot(times, data)
    plt.show()

if __name__ == '__main__':
    # toggles
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ex_id', type=str,
        help='Experiment timestamp (e.g. 20130318_105559)')
    parser.add_argument('data_type', type=str,
        help='Type of data to view (e.g. length_mm)')
    parser.add_argument('-f', '--fetch', action='store_true',
        help='Fetch data from phoenix')
    args = parser.parse_args()

    if args.fetch:
        # if data not already local, run this command:
        pull_plate_from_phoenix(args.ex_id)

    plot_all(args.ex_id, args.data_type)
