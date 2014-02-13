#!/usr/bin/env python
# WARNING THIS IS NONFUNCTIONAL CODE UNDER CONSTRUCTION
'''
Filename: compute_N_for_plate.py
Description:
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_directory)

# nonstandard imports
from Import.Code.experiment_index import Experiment_Attribute_Index
from Shared.Code.Database.mongo_retrieve import mongo_query, pull_data_type_for_blob

def choose_ex_id(key='purpose', value='N2_aging'):
    ei = Experiment_Attribute_Index()
    return ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value)

def determine_threshold_for_ex_id(ex_id):
    """
    """
    print ex_id
    docs = mongo_query({'ex_id':ex_id, 'data_type': 'metadata'}, {'blob_id':1, 'size_median':1})
    benchmark_blob, benchmark_median = '', 1000000
    for doc in docs:
        print doc['blob_id'], doc['size_median']
        if doc['size_median'] < benchmark_median:
            benchmark_median = doc['size_median']
            benchmark_blob = doc['blob_id']
    print 'benchmark:', benchmark_blob
    print 'median', benchmark_median
    sizes = pull_data_type_for_blob(blob_id=benchmark_blob, data_type='size_raw')['data'].values()
    benchmark_mean = np.mean(sizes)
    benchmark_std = np.std(sizes)
    print 'mean', benchmark_mean
    print 'std', benchmark_std
    print 'warning... the smallest blob might not be good'

def create_sizecheck_file(search_dir, size_threshold=300):
    import glob
    if '/' != search_dir[-1]:
        search_dir += '/'
    files = glob.glob('{path}*.blobs'.format(path=search_dir))
    if len(files) <1:
        print 'Warning: {path}\n may not be the correct directory. no blobs files found'.format(path=search_dir)
    save_name = '{path}size_threshold{T}.sizecheck'.format(path=search_dir, T=size_threshold)
    if os.path.exists(save_name):
        return save_name
    all_outlines = []
    for f in files:
        new_outlines = pull_sizes_from_blobs_file(filename=f, size_threshold=size_threshold)
        all_outlines += new_outlines
    with open(save_name, 'w') as f:
        for outline_data in all_outlines:
            line = [str(o) for o in outline_data]
            f.write(', '.join(line) + '\n')
    return save_name



def pull_sizes_from_blobs_file(filename, size_threshold):
    """
    """
    def test_sizes(sizes, size_threshold=size_threshold):
        return np.median(sizes) >= size_threshold

    def make_local_id(local_id):
        # local blob ids in the database are always length 5.
        #if this one is short, pad front with '0's
        for i in range(5):
            if len(local_id) < 5:
                local_id = '0' + local_id
        return local_id

    blob_size_check = []
    with open(filename, 'r') as f:
        local_id, start_t, end_t, if_good = None, None, None, None
        sizes = []
        for line in f:
            if line[0] == '%':
                if local_id:                    
                    if_good = test_sizes(sizes)
                    blob_size_check.append((local_id, start_t, end_t, if_good))
                local_id = make_local_id(line[1:].strip())
                sizes = []
                start_t = None
                end_t = None
            else:
                cols = line.split()
                time, size = float(cols[1]), int(cols[4])
                if not start_t:
                    start_t = time
                end_t = time
                sizes.append(size)

        if_good = test_sizes(sizes)                
        blob_size_check.append((local_id, start_t, end_t, if_good))
    return blob_size_check

def compute_N(ex_id):
    """
    """
    # 
    p = '/home/projects/worm_movement/Data/MWT_RawData/' + ex_id + '/'
    sizecheck_file = create_sizecheck_file(p)
    all_N = [0 for _ in xrange(3601)]
    good_N = [0 for _ in xrange(3601)]

    # parse sizecheck file into timeseries
    with open(sizecheck_file, 'r') as f:
        for line in f:
            cols = line.split(',')
            if cols[1].strip() == 'None' or cols[2].strip() == 'None':
                continue
            
            start, end = int(float(cols[1])), int(float(cols[2]))
            isGood = True
            if cols[3].strip() == 'False':
                isGood = False
            for i in xrange(start, end):
                all_N[i] += 1
            if isGood:
                for i in xrange(start, end):
                    good_N[i] += 1

    times = range(3601)
    return times, all_N, good_N

if __name__ == '__main__':
    ex_ids = choose_ex_id()
    times, all_N, good_N = compute_N(ex_id=ex_ids[1])
    plt.figure()
    plt.plot(times, all_N, color='blue')
    plt.plot(times, good_N, color='green')
    plt.show()

