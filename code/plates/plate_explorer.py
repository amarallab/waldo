
# coding: utf-8

# In[50]:

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.abspath('.')
CODES = os.path.abspath(os.path.join(HERE, '..'))
SHARED = os.path.join(CODES, 'shared')
print SHARED
sys.path.append(SHARED)
sys.path.append(CODES)

from wio.file_manager import read_table, ensure_dir_exists, get_good_blobs, get_timeseries
from settings.local import LOGISTICS
from metrics.measurement_switchboard import pull_blob_data

# TODO: eliminate redundency in function.
def pull_plate_from_phoenix(ex_id, dataset, pull=True):
    ''' pulls all plate-level and worm-level data from phoenix to local version of waldo.'''
    p_address = 'phoenix.research.northwestern.edu'
    # source
    p1 = '/home/projects/worm_movement/waldo/data/plates/{dset}/timeseries/{eid}*'.format(eid=ex_id, dset=dataset)
    p2 = '/home/projects/worm_movement/waldo/data/plates/{dset}/percentiles/{eid}*'.format(eid=ex_id, dset=dataset)
    w1 = '/home/projects/worm_movement/waldo/data/worms/{eid}'.format(eid=ex_id)
    # destination
    d_p1 = '{p_dir}timeseries'.format(p_dir=LOGISTICS['plates'])
    d_p2 = '{p_dir}percentils'.format(p_dir=LOGISTICS['plates'])
    d_w1 = '{w_dir}'.format(w_dir=LOGISTICS['worms'])
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
        blob_ids = get_good_blobs(ex_id=ex_id, data_type=data_type)
    print '{N} blob_ids found'.format(N=len(blob_ids))
    for blob_id in blob_ids:
        times, data = pull_blob_data(blob_id, metric=data_type)
        if times != None and len(times):
            print blob_id
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
    dataset = 'N2_aging'
    #ex_id = '20130318_105559'
    ex_id = '20130318_131111'    
    data_type = 'length_mm'
    
    # if data not already local, run this command:
    #pull_plate_from_phoenix(ex_id, dataset)

    plot_all(ex_id, data_type)



