#!/usr/bin/env python

'''
Filename: measurement_suite.py
Description: calls all subfunctions required to make all the types of measurements.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import numpy as np
from itertools import izip

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../')
sys.path.append(SHARED_DIR)

# nonstandard imports




#from wio.export_data import write_full_plate_timeseries
# add angle over distance


if __name__ == '__main__':
    if len(sys.argv) < 2:
        #bi = '00000000_000001_00001'
        #bi = '00000000_000001_00008'
        #measure_all(blob_id=bi)
        #write_plate_timeseries_set(ex_id='00000000_000001')
        write_plate_percentiles(ex_id='00000000_000001', blob_ids=['00000000_000001_00001', 
                                                                   '00000000_000001_00002', 
                                                                   '00000000_000001_00003', 
                                                                   '00000000_000001_00004', 
                                                                   '00000000_000001_00005', 
                                                                   '00000000_000001_00006', 
                                                                   '00000000_000001_00007', 
                                                                   '00000000_000001_00008'])

    else:
        ex_ids = sys.argv[1:]
        for ex_id in ex_ids[:]:
            print 'searching for blobs for', ex_id
            measure_all_for_ex_id(ex_id)
