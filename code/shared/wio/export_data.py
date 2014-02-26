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
from itertools import izip

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../')
sys.path.append(SHARED_DIR)

# nonstandard imports
from wormmetrics.measurement_switchboard import pull_blob_data
from database.mongo_retrieve import mongo_query
from file_manager import EXPORT_PATH, manage_save_path, get_blob_ids
from settings.local import LOGISTICS

PLATE_DIR = LOGISTICS['data'] + 'plate_timeseries'

#from file_manager import manage_save_path, get_blob_ids, EXPORT_PATH

'''
def export_blob_percentiles_by_ex_id(ex_id, out_dir=EXPORT_PATH, path_tag='', verbose=True, **kwargs):
    """
    write a json for an ex_id in which each blob has a dictionary of measurements containing
    a list with the 10, 20, 30, ... 80, 90 th percentiles.

    :param ex_id: the ex_id to write a json for.
    :param out_dir: the directory to write to.
    """
    save_name = manage_save_path(out_dir, path_tag, ID=ex_id, data_type='percentiles')
    blob_ids = get_blob_ids(query={'ex_id': ex_id, 'data_type': 'smoothed_spine'}, **kwargs)
    blob_data = {}
    N = len(blob_ids)
    for i, blob_id in enumerate(blob_ids):
        if verbose:
            print 'calculating metrics for {bID} ({i}/{N})'.format(bID=blob_id, i=i, N=N)
        blob_data[blob_id] = {}
        try:
            for metric, data in sb.pull_all_for_blob_id(blob_id, **kwargs).iteritems():
                #print len(data)
                blob_data[blob_id][metric] = [scoreatpercentile(data, i) for i in xrange(10, 100, 10)]
        except Exception as e:
            print blob_id, 'not working', e

    json.dump(blob_data, open(save_name, 'w'), indent=4, sort_keys=True)
'''

def export_index_files(props=['age', 'source-camera', 'label'], dataset='N2_aging',
                       store_blobs=True, out_dir=EXPORT_PATH, tag='', **kwargs):
    """
    writes a json with a dictionary of ex_ids and the value of a correspoinding property
    :param prop: the property to be exported (key string for database documents)
    :param out_dir: the directory in which to save the json.
    """

    data = {}
    if store_blobs:
        id_type = 'blob_id'
    else:
        id_type = 'ex_id'

    for e in mongo_query({'dataset': dataset, 'data_type': 'smoothed_spine'}, 
                         {'data': 0}, **kwargs):
        if e[id_type] not in data:
            data[e[id_type]] = {}
        for prop in props:
            data[e[id_type]][prop] = e.get(prop, 'none')

    save_name = '{d}/index{tag}.json'.format(d=out_dir, tag=tag)
    json.dump(data, open(save_name, 'w'), indent=4, sort_keys=True)

    save_name = '{d}/index{tag}.txt'.format(d=out_dir, tag=tag)
    with open(save_name, 'w') as f:
        headers = [id_type] + props
        f.write(',\t'.join(headers))
        for dID in sorted(data):
            line = [dID] + [data[dID][i] for i in props]
            f.write(','.join(map(str, line)) + '\n')

def write_full_plate_timeseries(ex_id, metric='cent_speed_mm', path_tag='', 
                                out_dir=PLATE_DIR, save_name=None, 
                                as_json=False, blob_ids=[], **kwargs):
                                
    # manage save path + name
    if not save_name:
        save_name = manage_save_path(out_dir=out_dir, path_tag=path_tag, 
                                     ID=ex_id, data_type=metric)
    if not blob_ids:
        # make list of all blobs
        blob_ids = get_blob_ids(query={'ex_id': ex_id, 'data_type':'spine'}, **kwargs)
    # compile a dict containing values for all blobs. data binned every 10th second.
    data_dict = {}
    for blob_id in blob_ids:
        # calculate metric for blob, skip if empty list returned        
        times, data = pull_blob_data(blob_id, metric=metric, **kwargs)
        if len(data) == 0:
            continue
        # store blob values in comiled dict.
        for t, value in izip(times, data):
            new_key = ('%.1f' % float(round(t, ndigits=1)))
            if new_key not in data_dict:
                data_dict[new_key] = []
            data_dict[new_key].append(value)

    # if empty, write anyway, but print warning
    if len(data_dict) == 0:
        print ex_id, 'has no data'
    if as_json:
        json.dump(data_dict, open(save_name, 'w'), indent=4, sort_keys=True)
    elif len(data_dict) > 1:
        times_sorted = sorted([(float(t), t) for t in data_dict])
        with open(save_name+'.dat', 'w') as f:
            for (tf, t) in times_sorted:
                line = '{t},{l}'.format(t=t, l=','.join(map(str, data_dict[t])))
                f.write(line + '\n')


def pull_blob_timeseires_for_ex_id(ex_id, data_type, out_dir=EXPORT_PATH, path_tag='',
                                   **kwargs):
    ''' saves a json with dictionary of all blobs for an ex_id.
    each blob is a dict with two lists: 'time' and 'data' that contain the time series.
    this function will create a unique name to identify data.

    blob_id - blob identifyer (str)
    data_type - type of data to save (str)
    out_dir - path to the directory to save the file (subdirectires will be created)
    path_tag - used to extend the name of the path, if desired.
    '''

    # manage save path + name
    if not save_name:
        save_name = mangage_save_path(out_dir=out_dir, path_tag=path_tag, ID=ex_id, data_type=data_type)
    # get all blob ids
    blob_ids = get_blob_ids(query={'ex_id': ex_id, 'data_type': 'smoothed_spine'}, **kwargs)
    print len(blob_ids), 'blob ids found'

    blob_data = {}
    for blob_id in blob_ids:
        blob_data[blob_id] = {}
        try:
            times, data = pull_blob_data(blob_id, metric=data_type, **kwargs)
            blob_data[blob_id] = {'time':times, 'data':data}
        except Exception as e:
            print 'Exception for {bi}'.format(bi=blob_id)
            print e
    json.dump(blob_data, open(save_name, 'w'), indent=4, sort_keys=True)


def pull_single_blob_timeseries(blob_id, data_type, out_dir=EXPORT_PATH, path_tag='', savename=None, **kwargs):
    ''' saves a json with two lists (time and data) for a particular blob_id and data type

    blob_id - blob identifyer (str)
    data_type - type of data to save (str)
    out_dir - path to the directory to save the file (subdirectires will be created)
    path_tag - used to extend the name of the path, if desired.
    savename - bypass all path creation steps and save file with this name.
    '''
    times, data = pull_blob_data(blob_id, metric=data_type, **kwargs)
    if not savename:
        save_name = mangage_save_path(out_dir=out_dir, path_tag=path_tag, ID=blob_id, data_type=data_type)
    json.dump({'time':times, 'data':data}, open(savename, 'w'), indent=4, sort_keys=True)


if __name__ == '__main__':
    # INDEX FILE EXAMPLES
    #export_index_files(props=['label', 'age'], tag='_')
    #export_index_files(props=['age'], tag='_N2ages', store_blobs=False)
    export_index_files(props=['midline_median'], tag='_N2ages_midlinemedian', store_blobs=True)
    #export_index_files(props=['duration'], tag='_duration')
    #export_ex_id_index(prop='age')
    #export_ex_id_index(prop='source-camera')

    # WRITE FULL PLATE TIMESERIES EXAMPLE
    ex_id = '20130320_102312'
    write_full_plate_timeseries(ex_id, save_name='test{id}.txt'.format(id=ex_id), **kwargs)

    # WORM TIME-SERIES EXAMPLES
    # data toggles
    ex_ids = get_ex_ids({'dataset':'N2_aging', 'data_type':'smoothed_spine'})
    print ex_ids
    #ex_ids = ['20130319_134739'] #, '20130321_144656', '20130320_140644']
    #data_type = 'speed_along_bl'
    #data_type = 'centroid_speed_bl'
    data_type = 'smooth_length'

    # save toggles
    dataset = 'segment_test'
    dataset = 'length'
    out_dir = '{path}{dset}-{dt}/'.format(path=EXPORT_PATH, dset=dataset, dt=data_type)
    for ex_id in ex_ids:
        pull_data_for_ex_id(ex_id, data_type, out_dir=out_dir)

    # to export percentiles
    ex_id = '00000000_000001'
    #export_blob_percentiles_by_ex_id(ex_id)

