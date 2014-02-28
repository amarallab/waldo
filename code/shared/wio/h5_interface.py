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
    
def write_h5_timeseries_base(h5_file, h5_path, times, data):
    times = np.array(times) 
    data = np.array(data)
    #print data
    with h5py.File(h5_file, 'w') as f:
        grp = f.create_group(h5_path)        
        print 'times', times.shape
        print 'data', data.shape        
        grp.create_dataset(name='time',                           
                           shape=times.shape,
                           dtype=float,
                           data=times,
                           chunks=True,
                           compression='lzf')
        grp.create_dataset(name='data',                           
                           shape=data.shape,
                           dtype=float,
                           data=data,
                           chunks=True,
                           compression='lzf')
        
                                                               
def read_h5_timeseries_base(h5_file, h5_path):
    times, data = [], []
    path_parts = map(unicode, h5_path.split('/'))
    #print path_parts
    with h5py.File(h5_file, 'r') as f:
        #times_path = '{p}/time'.format(p=h5_path.rstrip('/'))
        #data_path = '{p}/data'.format(p=h5_path.rstrip('/'))        
        part1, parts = path_parts[0], path_parts[1:]
        if part1 in f.keys():
            grp = f[part1]    
            for i, p in enumerate(parts):
                #print i, p
                if p in grp.keys():
                    grp = grp[p]
                
            times = np.array(grp['time'])
            data = np.array(grp['data'])
            #data = np.array(f[data_path])            
                    
        #print grp.keys()
        #print times_path
        #print data_path
        

    return times, data

def delete_h5_dataset(h5_file, h5_path, times, data):
    pass
