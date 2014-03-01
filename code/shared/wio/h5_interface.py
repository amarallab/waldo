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
    #print data
    #if not os.path.isfile(h5_file):
    #    h5py.File(h5_file, 'w-')
            
    with h5py.File(h5_file, 'w') as f:                
        #grp = f.create_group(h5_path)        
        #print 'times', times.shape
        #print 'data', data.shape        
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
    #path_parts = map(unicode, h5_path.split('/'))
    #print path_parts
    with h5py.File(h5_file, 'r') as f:
        #times_path = '{p}/time'.format(p=h5_path.rstrip('/'))
        #data_path = '{p}/data'.format(p=h5_path.rstrip('/'))
        #print f.keys()
        '''
        part1, parts = path_parts[0], path_parts[1:]
        if part1 in f.keys():
            grp = f[part1]
            print grp.keys()
            for i, p in enumerate(parts):
                #print i, p
                if p in grp.keys():
                    grp = grp[p]
        '''     
        times = np.array(f['time'])
        data = np.array(f['data'])
            #data = np.array(f[data_path])            
        #print grp.keys()
        #print times_path
        #print data_path    
    return times, data

def delete_h5_dataset(h5_file, h5_path, times, data):
    pass



# main is purely for testing purposes
if __name__ == '__main__':
    from file_manager import get_timeseries, format_h5_path
    bID = '00000000_000001_00001'
    data_type = 'encoded_outline'
    data_type = 'spine_rough'
    h5_file = format_h5_path(blob_id=bID, data_type=data_type,
                                      h5_dir='./')
    times, data = get_timeseries(blob_id=bID, data_type=data_type)

    print NA_spine
    
    times = np.array(times)
    data = np.array(data)
    data = np.array(dat)
    #data = np.array(data, dtype=str)
    print data.dtype
    print times[:1]
    print data[:1]    
    write_h5_timeseries_base(h5_file, times, data)    
    write_h5_timeseries_base(h5_file, times, data)
    times, data = read_h5_timeseries_base(h5_file)
    print times[:1]
    print data[:1]    
    #x, y, l, o = zip(*data)
   
