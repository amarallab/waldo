#!/usr/bin/env python

'''
Filename: h5_interface.py

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
from itertools import izip
import h5py
import numpy as np
'''
# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)
'''

OPTIONS = {}

def write_h5_outlines(h5_file, h5_path, times, data):
    pass
    
def write_h5_timeseries_base(h5_file, times, data):
    times = np.array(times) 
    data = np.array(data)
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset(name='time',                           
                           shape=times.shape,
                           dtype=times.dtype,
                           data=times,
                           chunks=True,
                           compression='lzf')
        f.create_dataset(name='data',                           
                           shape=data.shape,
                           dtype=data.dtype,
                           data=data,
                           chunks=True,
                           compression='lzf')
                
                                                               
def read_h5_timeseries_base(h5_file):
    times, data = [], []    
    with h5py.File(h5_file, 'r') as f:
        times = np.array(f['time'])
        data = np.array(f['data'])
    return times, data

'''
def write_h5_timeseries_base1(h5_file, h5_path, times, data):
    times = np.array(times) 
    data = np.array(data)
    with h5py.File(h5_file, 'w') as f:
        print 'writing', h5_path, h5_path in f
        if h5_path in f:
            grp = f[h5_path]
        else:
            grp = f.create_group(h5_path)
        grp.create_dataset(name='time',                           
                           shape=times.shape,
                           dtype=times.dtype,
                           data=times,
                           chunks=True,
                           compression='lzf')
        grp.create_dataset(name='data',                           
                           shape=data.shape,
                           dtype=data.dtype,
                           data=data,
                           chunks=True,
                           compression='lzf')
                                                                               
def read_h5_timeseries_base1(h5_file, h5_path):
    times, data = [], []
    print 'path', h5_path    
    path_parts = map(unicode, h5_path.split('/'))
    with h5py.File(h5_file, 'r') as f:
        print 'read', path_parts[0], path_parts[0] in f        
        print 'read', h5_path, h5_path in f
        if h5_path in f:
            grp = f[h5_path]
            times = np.array(grp['time'])
            data = np.array(grp['data'])
    print len(times), len(data)
    return times, data
'''
def delete_h5_dataset(h5_file, h5_path, times, data):
    pass

# main is purely for testing purposes
if __name__ == '__main__':
    from file_manager import get_timeseries, format_h5_path
    bID = '00000000_000001_00001'
    data_type = 'encoded_outline'
    #data_type = 'spine_rough'
    h5_file, h5_path = format_h5_path(blob_id=bID, data_type=data_type,
                                      h5_dir='./')
    times, data = get_timeseries(blob_id=bID, data_type=data_type)
    
    times = np.array(times)
    #data = np.array(data)
    data = np.array(data, dtype=str)
    print data.dtype
    print times[:1]
    print data[:1]    
    write_h5_timeseries_base1(h5_file, h5_path, times, data)
    write_h5_timeseries_base1(h5_file, 'A/A', times, data)        
    times, data = read_h5_timeseries_base1(h5_file, h5_path)
    print times[:1]
    print data[:1]    
    #x, y, l, o = zip(*data)
   
