#!/usr/bin/env python

'''
Filename: plate_utilities.py
Description:
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import datetime
from itertools import izip
import numpy as np
import json
import glob
import shutil

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from file_manager import ensure_dir_exists, PLATE_DIR, DSET_DIR, \
     format_filename, format_dirctory, get_timeseries, get_dset
from annotation.experiment_index import Experiment_Attribute_Index

def remove_plate_files(ex_id, file_tags):
    dataset = get_dset(ex_id)

    for tag in file_tags:
        print tag
        search_dir = os.path.abspath(format_dirctory(ID=ex_id, ID_type='plate',
                                                     dataset=dataset,
                                                     tag=tag))
        search = '{d}/{eID}*'.format(d=search_dir, eID=ex_id)
        for rfile in glob.iglob(search):
            print 'removing:', rfile
            os.remove(rfile)

def get_plate_files(dataset, data_type, tag='timeseries', path=None):
    if not path:
        path = format_dirctory(ID_type='plate',
                               dataset=dataset,
                               tag=tag)
    search = '{path}/*'.format(path=path.rstrip('/'))
    ex_ids, file_paths = [], []
    for file_path in glob.glob(search):
        if os.path.isfile(file_path):
            ex_id = file_path.split('/')[-1].split('-' +data_type)[0]
            # one small check before storing
            if len(ex_id.split('_')) == 2:
                ex_ids.append(ex_id)
                file_paths.append(file_path)
    return ex_ids, file_paths

def ex_id_to_datetime(ex_id):
    ''' converts an experiment id to a datetime object '''     
    parts = ex_id.split('_')
    if len(parts) != 2:
        print 'Error: something is off with this ex_id', ex_id
        return None
    ymd, hms = parts
    year, month, day = map(int, [ymd[:4], ymd[4:6], ymd[6:]])
    h, m, s = map(int, [hms[:2], hms[2:-2], hms[-2:]])
    return datetime.datetime(year, month, day, h, m, s)

    
def format_plate_summary_name(ex_id, sum_type, dataset, data_type, path):
    filename = format_filename(ID=ex_id, ID_type='plate',
                               file_tag=sum_type,
                               dset=dataset,
                               data_type=data_type,
                               file_dir=path,
                               file_type='json')
    return filename


 
def format_dset_summary_name(data_type, dataset, sum_type, ID= None, dset_dir=None):
    if not ID:        
        ID=dataset
    filename = format_filename(ID=ID, ID_type='dset',
                               data_type=data_type,
                               file_tag=sum_type,
                               dset=dataset,
                               file_dir=dset_dir,
                               file_type='json')
    
    return filename



def write_dset_summary(data, data_type, dataset, sum_type, ID=None, dset_dir=None):
    filename = format_dset_summary_name(data_type, dataset, sum_type, ID, dset_dir)
    print filename
    json.dump(data, open(filename, 'w'))

def read_dset_summary(data_type, dataset, sum_type='basic', ID=None, dset_dir=None):
    filename = format_dset_summary_name(data_type, dataset, sum_type, ID, dset_dir)
    if os.path.isfile(filename):
        return json.load(open(filename, 'r'))
    else:
        return {}

def write_plate_summary(data, ex_id, sum_type, dataset, data_type, path=None):
    filename = format_plate_summary_name(ex_id, sum_type, dataset, data_type, path)
    json.dump(data, open(filename, 'w'))

def read_plate_summary(sum_type, dataset, data_type, path=None):
    filename = format_plate_summary_name(ex_id, sum_type, dataset, data_type, path)
    if os.path.isfile(filename):
        return json.load(open(filename, 'r'))
    return None

def read_plate_timeseries(ex_id, dataset, data_type, tag='timeseries'):
    times, data = get_timeseries(ID=ex_id, 
                                 data_type=data_type,
                                 ID_type='p',                                   
                                 dset=dataset,
                                 file_tag=tag)
    #print len(times), len(data)
    return times, data

def organize_plate_metadata(ex_id):
    ei = Experiment_Attribute_Index()
    m = ei.return_attributes_for_ex_id(ex_id)
    if m == None:
        m = {}
    label = m.get('label', 'label')
    sub_label = m.get('sublabel', 'set')
    sub_label = '{sl}'.format(sl=sub_label)
    pID = m.get('plate-id', 'set B')
    day = m.get('age', 'A0')

    recording_time = ex_id
    plating_time = m.get('l1-arrest', None)
    #print 'plated at:', plating_time
    #print 'recorded at:', recording_time
    hours = 0
    if recording_time and plating_time:
        t0 = ex_id_to_datetime(plating_time)
        t1 = ex_id_to_datetime(recording_time) 
        hours = (t1 - t0).total_seconds()/3600.
    #age = '{et} - {pt}'.format(et=recording_time, pt=plating_time)
    #for i in m:
    #    print i
    return hours, label, sub_label, pID, day

def return_flattened_plate_timeseries(ex_id, dataset, data_type):
    """
    """    
    #times, data = parse_plate_timeseries_txt_file(dfile)
    times, data = read_plate_timeseries(ex_id, dataset, data_type, tag='timeseries')
    #times, data = parse_plate_timeseries_txt_file(dfile)
    flat_data = []
    if times == None:
        return []
    if not len(times):
        return []
    for i, t_bin in enumerate(data):
        flat_data.extend(list(t_bin))
    # take data out of bins and remove nan values
    flat_data = np.array(flat_data)
    N_w_nan = len(flat_data)
    flat_data = flat_data[np.logical_not(np.isnan(flat_data))]
    N_wo_nan = len(flat_data)
    if N_wo_nan != N_wo_nan:
        print '{N} nans removed'.format(N=N_w_nan-N_wo_nan)
    return flat_data

if __name__ == '__main__':
    dataset = 'disease_models'
    data_type = 'cent_speed_bl'
    #show_timeseries_options()
    ex_ids, file_paths = get_ex_id_files(dataset, data_type)
