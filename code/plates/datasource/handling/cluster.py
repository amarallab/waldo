# -*- coding: utf-8 -*-
"""
FILL ME IN
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import sys
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.abspath('.')
CODES = os.path.abspath(os.path.join(HERE, '..'))
SHARED = os.path.join(CODES, 'shared')
#print SHARED
sys.path.append(SHARED)
sys.path.append(CODES)

from wio.file_manager import read_table, ensure_dir_exists, get_good_blobs, get_timeseries, get_dset
from settings.local import LOGISTICS
#from metrics.measurement_switchboard import pull_blob_data

# TODO: eliminate redundency in function.
def pull(ex_id, dry_run=False):
    ''' pulls all plate-level and worm-level data from phoenix to local version of waldo.'''
    dataset = get_dset(ex_id)
    p_address = 'phoenix.research.northwestern.edu'
    # source
    p1 = '/home/projects/worm_movement/waldo/data/plates/{dset}/timeseries/{eid}*'.format(eid=ex_id, dset=dataset)
    p2 = '/home/projects/worm_movement/waldo/data/plates/{dset}/percentiles/{eid}*'.format(eid=ex_id, dset=dataset)
    w1 = '/home/projects/worm_movement/waldo/data/worms/{eid}'.format(eid=ex_id)
    # destination
    d_p1 = '{p_dir}timeseries'.format(p_dir=LOGISTICS['plates'])
    d_p2 = '{p_dir}percentiles'.format(p_dir=LOGISTICS['plates'])
    d_w1 = '{w_dir}'.format(w_dir=LOGISTICS['worms'])
    # make sure destination exists
    for d in [d_p1, d_p2, d_w1]:
        print(d)
        ensure_dir_exists(d)
    # commands
    cmd_p1 = 'scp -v {address}:{cmd} {dest}'.format(address=p_address, cmd=p1, dest=d_p1)
    cmd_p2 = 'scp -v {address}:{cmd} {dest}'.format(address=p_address, cmd=p2, dest=d_p2)
    cmd_w1 = 'scp -rv {address}:{cmd} {dest}'.format(address=p_address, cmd=w1, dest=d_w1)
    # run all the commands
    for cmd in [cmd_p1, cmd_p2, cmd_w1]:
        print('\n', cmd, '\n')
        if not dry_run:
            os.system(cmd)
