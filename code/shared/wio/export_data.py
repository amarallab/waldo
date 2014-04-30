
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
import pandas as pd

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../')
sys.path.append(SHARED_DIR)

# nonstandard imports
from metrics.measurement_switchboard import pull_blob_data
from file_manager import EXPORT_PATH, manage_save_path, get_good_blobs
from file_manager import ensure_dir_exists, get_timeseries
from annotation.experiment_index import Experiment_Attribute_Index2

def create_direcotry_structure(dataset, path=EXPORT_PATH):
    ei = Experiment_Attribute_Index2(dataset)
    ex_ids = list(ei.index)
    path = os.path.abspath(path)
    columns = ['name', 'pixels-per-mm']
    print columns
    
    ex_id_data = []
    index = []
    for ex_id, name in ei['name'].iteritems(): #ex_ids:
        name = name.strip().replace(' ', '_')
        save_dir = '{p}/{dset}/{e}-{n}'.format(p=path, dset=dataset, e=ex_id, n=name)
        success = write_worm_exports_for_plate(ex_id, save_dir)
        if success:
            print ex_id, name, 'sucess'
            #print ei.loc[ex_id][columns].head()
            index.append(ex_id)
            ex_id_data.append(ei.loc[ex_id][columns])
            #second = ei.loc[ex_id][columns]
            #second.name = '2'
            #ex_id_data.append(second)            
    ex_id_index = pd.concat(ex_id_data, axis=1).T
    #print 'col', ex_id_index.columns
    #print 'index', ex_id_index.index        
    index_name = '{p}/{dset}/index.csv'.format(p=path, dset=dataset)
    print index_name    
    print ex_id_index.head()
    ex_id_index.to_csv(index_name)
            
def write_worm_exports_for_plate(ex_id, save_dir):
    blobs = get_good_blobs(ex_id, key = 'xy')
    #print len(blobs), 'blobs found'
    if not len(blobs):
        return False
    ensure_dir_exists(save_dir)
    for blob_id in blobs:
        success = write_worm_export(blob_id, save_dir)
        #if success:
        #    break
    else:
        return False
    return True
               
def write_worm_export(blob_id, save_dir):
    #df = pd.DataFrame()
    print blob_id
    dtypes = ['xy_raw', 'xy']
    mtypes = ['cent_speed_bl', 'cent_speed_mm']

    times, data = get_timeseries(blob_id, data_type='xy_raw')
    if times != None and len(times) > 0:
        xyraw = pd.DataFrame(data, index=times, columns=['x', 'y'])    
        xyraw.index.name = 'time (s)'
        savename = '{d}/raw_xy-{ID}.csv'.format(d=save_dir, ID=blob_id)
        xyraw.to_csv(savename)
        
    times, data = get_timeseries(blob_id, data_type='xy')
    if times != None and len(times) > 0:
        xydata = pd.DataFrame(data, index=times, columns=['x', 'y'])    
    else:
        return False
        
    times, data = pull_blob_data(blob_id, metric='cent_speed_bl')        
    if times != None and len(times) > 0:
        bldata = pd.DataFrame(data, index=times, columns=['blps'])
    else:
        return False

    times, data = pull_blob_data(blob_id, metric='cent_speed_mm')        
    if times != None and len(times) > 0:
        mmdata = pd.DataFrame(data, index=times, columns=['mmps'])    
    else:
        return False

    combined = pd.concat([xydata, bldata, mmdata], axis=1)
    print combined.head()
    savename = '{d}/worm_track-{ID}.csv'.format(d=save_dir, ID=blob_id)
    print savename
    combined.index.name = 'time (s)'
    combined.to_csv(savename)
    return True
'''       
def write_full_plate_timeseries(ex_id, metric='cent_speed_mm', path_tag='', 
                                out_dir=PLATE_DIR, save_name=None, 
                                as_json=False, blob_ids=[], **kwargs):
    # manage save path + name
    if not save_name:
        save_name = manage_save_path(out_dir=out_dir, path_tag=path_tag, 
                                     ID=ex_id, data_type=metric)
    if not blob_ids:
        # make list of all blobs
        blob_ids = get_good_blobs(ex_id)
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
'''

def pull_blob_timeseires_for_ex_id(ex_id, data_type, out_dir=EXPORT_PATH, path_tag='',
                                   **kwargs):
    ''' saves a json with dictionary of all blobs for an ex_id.
    each blob is a dict with two lists: 'time' and 'data' that
    contain the time series. this function will create a unique
    name to identify data.
    
    blob_id - blob identifyer (str)
    data_type - type of data to save (str)
    out_dir - path to the directory to save the file (subdirectires will be created)
    path_tag - used to extend the name of the path, if desired.
    '''
    # manage save path + name
    if not save_name:
        save_name = mangage_save_path(out_dir=out_dir, path_tag=path_tag,
                                      ID=ex_id, data_type=data_type)
    # get all blob ids
    blob_ids = get_blob_ids(query={'ex_id': ex_id, 'data_type': 'smoothed_spine'},
                             **kwargs)
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

def pull_single_blob_timeseries(blob_id, data_type, out_dir=EXPORT_PATH,
                                path_tag='', savename=None, **kwargs):
    ''' saves a json with two lists (time and data) for a particular
    blob_id and data type

    blob_id - blob identifyer (str)
    data_type - type of data to save (str)
    out_dir - path to the directory to save the file (subdirectires will be created)
    path_tag - used to extend the name of the path, if desired.
    savename - bypass all path creation steps and save file with this name.
    '''
    times, data = pull_blob_data(blob_id, metric=data_type, **kwargs)
    if not savename:
        save_name = mangage_save_path(out_dir=out_dir, path_tag=path_tag,
                                      ID=blob_id, data_type=data_type)
    json.dump({'time':times, 'data':data}, open(savename, 'w'),
              indent=4, sort_keys=True)

if __name__ == '__main__':
    # INDEX FILE EXAMPLES
    dataset = 'N2_aging'
    create_direcotry_structure(dataset)
    

    '''
    #export_index_files(props=['label', 'age'], tag='_')
    #export_index_files(props=['age'], tag='_N2ages', store_blobs=False)
    export_index_files(props=['midline_median'], tag='_N2ages_midlinemedian',
                       store_blobs=True)
    #export_index_files(props=['duration'], tag='_duration')
    #export_ex_id_index(prop='age')
    #export_ex_id_index(prop='source-camera')

    # WRITE FULL PLATE TIMESERIES EXAMPLE
    ex_id = '20130320_102312'
    write_full_plate_timeseries(ex_id, save_name='test{id}.txt'.format(id=ex_id),
                                **kwargs)

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
    out_dir = '{path}{dset}-{dt}/'.format(path=EXPORT_PATH, dset=dataset,
                                          dt=data_type)
    for ex_id in ex_ids:
        pull_data_for_ex_id(ex_id, data_type, out_dir=out_dir)
    # to export percentiles
    ex_id = '00000000_000001'
    #export_blob_percentiles_by_ex_id(ex_id)
    '''
