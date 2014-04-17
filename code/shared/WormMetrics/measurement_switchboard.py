#!/usr/bin/env python

'''
Filename: measurement_switchboard.py
Description: 
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import math
from itertools import izip

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(HERE + '/../../')
CODE_DIR = os.path.abspath(HERE + '/../')
SHARED_DIR = CODE_DIR + 'shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.file_manager import get_metadata, get_timeseries
from compute_metrics import *

#STANDARD_MEASUREMENTS = ['length_mm', 'curve_w', 'cent_speed_bl']
FULL_SET = ['length_mm', 'width_mm', 'curve_w', 'cent_speed_bl', 'angle_change']

STANDARD_MEASUREMENTS = FULL_SET

# globals
SWITCHES = {'width': {'func': compute_width, 'units': ['mm', 'bl']},
             'size': {'func': compute_size, 'units': ['mm2']},
             'length': {'func': compute_length, 'units': ['mm']},
             'curve': {'func': compute_curvature, 'units': ['mm', 'bl', 'w'], 
                      'position':['head', 'mid', 'tail']},
             'speed_along': {'func': compute_speed_along, 'units':['mm', 'bl'], 
                             'position':['head', 'mid', 'tail']},
             'speed_prep': {'func': compute_speed_perp, 'units':['mm', 'bl'], 
                             'position':['head', 'mid', 'tail']},
             'cent_speed': {'func': compute_centroid_speed,
                            'units': ['bl', 'mm']},
             'angle_change':{'func': compute_angle_change, 'units':['rad', 'deg']}
                            }        

# **** main function of the module. 
def pull_blob_data(blob_id, metric, pixels_per_mm=0, pixels_per_bl=0, 
                   pixels_per_w=0, **kwargs):
    ''' returns a list of times and a list of data for a given blob_id and metric.

    This function chooses which program to call in order to calculate or retrieve
    the desired metric.
    '''
    # get the data before skips removed, scaled to units
    times, data, args = find_data(blob_id, metric, **kwargs)    
    # skips are a nuscence. for now, remove them permenantly    
    remove_skips = True
    if remove_skips:
        ntimes, ndata = [], []
        for t,v in izip(times, data):
            if isinstance(v, float):
                ntimes.append(t)
                ndata.append(v)
        times, data = ntimes, ndata
    if len(data) ==0:
        return [], []    
    # this block of code handles all scaling factor operations
    scaling_factor_type = args.get('units', '')
    # make sure we have appropriate scaling factor
    if not pixels_per_mm and str(scaling_factor_type) in ['mm', 'mm2']:        
        metadata = get_metadata(blob_id, **kwargs)
        pixels_per_mm = float(metadata.get('pixels-per-mm', 1.0))
        if pixels_per_mm == 1.0:
            print 'Warning, not getting appropriate pixels-per-mm for', blob_id
    if not pixels_per_bl and scaling_factor_type=='bl':        
        _, lengths = compute_length(blob_id, **kwargs)
        pixels_per_bl = np.median(lengths)
    if not pixels_per_w and scaling_factor_type=='bl':        
        _, widths = compute_width(blob_id, **kwargs)
        pixels_per_w = np.median(widths)

    # implement that scaling factor.
    if scaling_factor_type == 'mm':
        data = np.array(data)  / pixels_per_mm        
    if scaling_factor_type == 'mm2':
        data = np.array(data)  / (pixels_per_mm ** 2)        
    if scaling_factor_type == 'bl':
        data = np.array(data)  / pixels_per_bl        
    if scaling_factor_type == 'w':
        if pixels_per_w != 0.0:
            data = np.array(data)  / pixels_per_w        
        else:
            return [], []    
    if scaling_factor_type == 'deg':
        data = np.array(data) * 180.0 / np.pi
    return times, data

def find_data(blob_id, metric, **kwargs):
    ''' order of operations for locating data.
    '''
    # check if result already cached locally
    times, data = get_timeseries(blob_id, data_type=metric, search_db=False)
    #print 'not already cached'
    if times != None and data != None:
        return times, data, {}
    # check if it can be calculated
    metric_computation_function, args = switchboard(metric)
    # TODO: bulild in args.
    if metric_computation_function:
        #print 'computing metric', metric, 'using', metric_computation_function
        times, data = metric_computation_function(blob_id=blob_id, metric=metric, **kwargs)
        return times, data, args
    #print 'metric not found'
    # check if it is cached locally or in db.
    times, data = get_timeseries(blob_id, data_type=metric, **kwargs)
    return times, data, {}

def measure_matches_metric(measure_type, metric):
    numparts = len(measure_type.split('_'))
    split_name = metric.split('_')
    if numparts > split_name:
        return False
    measure_sized_metric = '_'.join(split_name[:numparts])
    return str(measure_type) == str(measure_sized_metric)

def switchboard(metric, switches=dict(SWITCHES)):
    """
    returns the function and kwargs for a given metric string.
    
    :param metric:
    :type metric: str
    :return: function, kwargs
    """
    args = {}
    metric = str(metric)
    for measure_type, arg_options in switches.iteritems():        
        # this copy is to prevent SWITCHES from being changed.
        arg_options = dict(arg_options)
        m_function = arg_options.pop('func', None)
        if measure_matches_metric(measure_type, metric):
            args = metric.split(measure_type)[-1].split('_')
            kwargs = {}
            for arg in args:
                found_arg_option = False
                for key, values in arg_options.iteritems():
                    if arg in values:
                        kwargs[key] = arg
                        found_arg_option = True
                if len(arg) > 0 and not found_arg_option:
                    print 'warning: argument option not found', arg
            return m_function, kwargs

    print 'the metric you specified, ({m}) could not be located'.format(m=metric)
    return False, {}


# TODO: is this really used? in it's current state, this is better off removed.
'''
def pull_metric_for_ex_id(ex_id, metric, verbose=False):
    # TODO make dependent on pull metric for blob_id
    # find which function you need for that metric and which blob_ids to include on the search
    blob_ids = get_blob_ids(query={'ex_id': ex_id, 'data_type': 'smoothed_spine'},
                            **kwargs)
    # go through every blob and pool the data of that type
    pooled_data = []
    for blob_id in blob_ids[:]:
        new_values = pull_blob_data(blob_id, metric)
        pooled_data += list(new_values)
    return pooled_data
'''
def list_all_metrics():
    all_metrics = []
    for measure_type, properties in SWITCHES.iteritems():
        all_metrics += properties['metrics']
    return all_metrics

# TODO: is this really used? in it's current state, this is better off removed.
'''
def pull_all_for_blob_id(blob_id, out_format='values', **kwargs):
    all_data = {}
    assert out_format in ['values', 'timedict']
    times, flags, _ = get_data(blob_id, data_type='flags', **kwargs)
    for metric_type, properties in SWITCHES.iteritems():
        for metric in properties['metrics']:
            if out_format == 'values':
                all_data[metric] = pull_blob_data(blob_id, metric=metric, **kwargs).values()
            else:
                all_data[metric] = pull_blob_data(blob_id, metric=metric,
                                                           **kwargs)
    return all_data
'''


if __name__ == '__main__':
    '''
    metric = 'speed_perp'
    metric = 'size_mm2'
    #metric = 'centroid_ang_ds'
    ex_id = '00000000_000001'
    #pull_metric_for_ex_id(ex_id, metric)
    for (m, data) in pull_all_for_ex_id(ex_id):
        print m, len(data)
    '''
    bID = '20130415_104153_00853'
    #bID='00000000_000001_00001'
    import time
    start = time.time()
    pull_blob_data(blob_id=bID, metric='length')
    dur = time.time() - start

    print '1', dur

