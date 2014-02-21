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

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
SHARED_DIR = CODE_DIR + '/shared/'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from wormmetrics.measurement_switchboard import pull_blob_data
from wio.file_manager import write_tmp_file, get_metadata

# add angle over distance
STANDARD_MEASUREMENTS = ['length_mm', 'curve_bl', 'cent_speed_bl']

def measure_all(blob_id, store_tmp=True,  measurements=STANDARD_MEASUREMENTS, **kwargs):
    """
    Arguments:
    - `blob_id`:
    - `**kwargs`:
    """    
    print 'Computing Standard Measurements'
    # prepare scaling factors to convert from pixels to units
    metadata = get_metadata(blob_id, **kwargs)
    lt, lengths = pull_blob_data(blob_id, metric='length', **kwargs)
    pixels_per_bl = float(np.median(lengths))
    pixels_per_mm = float(metadata.get('pixels-per-mm', 1.0))
    print '\tunits mm : {mm} | bl: {bl}'.format(bl=round(pixels_per_bl, ndigits=2),
                                               mm=round(pixels_per_mm, ndigits=2))
    # loop through standard set of measurements and write them
    for metric in measurements:        
        t, data = pull_blob_data(blob_id, metric=metric,
                                 pixels_per_bl=pixels_per_bl,
                                 pixels_per_mm=pixels_per_mm, 
                                 **kwargs)
        if store_tmp:
            write_tmp_file(blob_id, data_type=metric, data={'time':t, 'data':list(data)})
        print '\tstored {m}'.format(m=metric)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        bi = '00000000_000001_00001'
        bi = '00000000_000001_00008'
        measure_all(blob_id=bi)
    else:
        ex_ids = sys.argv[1:]
        for ex_id in ex_ids[:]:
            print 'searching for blobs for', ex_id
            measure_all_for_ex_id(ex_id)
