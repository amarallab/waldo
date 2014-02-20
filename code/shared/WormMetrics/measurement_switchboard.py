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
from wio.file_manager import get_blob_ids, get_data
from GeometricCalculations.distance import euclidean

# globals
# TODO: Figure out a way to build the switchboard from the functions without hard coding in acceptable_types
SWITCHES = {'width': {'func': compute_width, 'units': ['mm', 'bl']},
             'size': {'func': compute_size, 'units': ['mm2']},
             'length': {'func': compute_length, 'units': ['mm']},
             'curv': {'func': compute_curvature, 'units': ['mm', 'bl'], 
                      'position':['head', 'mid', 'tail']},
             'speed_along': {'func': compute_speed_along, 'units':['mm', 'bl'], 
                             'position':['head', 'mid', 'tail']},
             'speed_prep': {'func': compute_speed_perp, 'units':['mm', 'bl'], 
                             'position':['head', 'mid', 'tail']},
             'cent_speed': {'func': compute_centroid_speed,
                            'units': ['bl', 'mm']}}        

# **** main function of the module. 
def pull_blob_data(blob_id, metric, remove_skips=True, **kwargs):
    ''' returns a list of times and a list of data for a given blob_id and metric.

    This function chooses which program to call in order to calculate or retrieve
    the desired metric.
    '''
    metric_computation_function, args = switchboard(metric)
    if metric_computation_function:
        times, data = metric_computation_function(blob_id=blob_id, metric=metric, **kwargs)
    else:
        times, data, _ = get_data(blob_id, data_type=metric, **kwargs)['data']
    if remove_skips:
        ntimes, ndata = [], []
        for t,v in izip(times, data)
            if isinstance(v, float):
                ntimes.append(t)
                ndata.append(v)
        times, data = ntimes, ndata
    return times, data

def switchboard(metric):
    """
    searches through SWITCHES Global dictionary and returns the function used to calculate it.
    
    :param metric:
    :type metric: str
    :return: function to calculate the
    :type return: function
    """
    def measure_matches_metric(measure_type, metric):
        nubparts = len(measure_type.split('_'))
        split_name = metric.split('_')
        if numparts > split_name:
            return False
        measure_sized_metric = '_'.join(split_name[:numparts])
        return str(measure_type) == str(measure_sized_metric)

    args = {}
    metric = str(metric)
    for measure_type, arg_options in SWITCHES.iteritems():        
        m_function = arg_options.pop('func', None)
        if right_name(measure_type, metric):
            args = metric.split(measure_type)[-1].split('_')
            
            return m_function, args

    print 'the metric you specified, ({m}) could not be located'.format(m=measure)
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

# TODO? move remove flag functionality further upstream?
def pull_basic_data_type(blob_id, data_type, remove_flags=True, **kwargs):    
    ''' default option to remove flagged timepoints!'''
    times, data = get_data(blob_id, data_type=data_type, **kwargs)
    if remove_flags:
        times_f, all_flags = get_data(blob_id, data_type='flags', **kwargs)
        flags = consolidate_flags(all_flags)
        unflagged_timeseries = [(t, s) for (t, s, f) in izip(times, data, flags) if f]
        times, data = zip(*unflagged_timeseries)
    return times, data

# todo
def space_basic_data_type_linearly(times, values):
    return times, values

def compute_size(blob_id, **kwargs):    
    # size is a body shape property. if we've flagged the shape, don't measure size.
    return pull_basic_data_type(blob_id, data_type='size_raw',
                                remove_flags=True, **kwargs)

def compute_width(blob_id, **kwargs):
    # size is a body shape property. if we've flagged the shape, don't measure size.
    return pull_basic_data_type(blob_id, data_type='width50', 
                                remove_flags=True, **kwargs)

def compute_centroid_speed(blob_id, **kwargs):
    # speed is not a body shape.
    # TODO get xy_raw processed ahead of time.
    times, xy, _ = get_data(blob_id, data_type='xy_raw', **kwargs)    
    stimes, speeds = [], []
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        xy_future = xy[i + 1]
        xy_now = xy[i]
        ds = math.sqrt((xy_future[0] - xy_now[0]) ** 2 + (xy_future[1] - xy_now[1]) ** 2)
        stimes.append(t[i])
        speeds.append(ds/dt)
    return times, speeds

def length_of_spine(spine):
        dx, dy = map(np.diff, map(np.array, zip(*spine)))
        return np.sqrt(dx**2 + dy**2).sum()

def speed_along_spine(spine1, spine2, t_plus_dt, t, perpendicular, points='all'):
    '''
        this function takes a spine
        list of [[x1,y1], [x2,y2], ...]
        and returns the average speed along (or perpendicular to) the spine
    '''
    dt = t_plus_dt - t
    assert dt > 0, 'times are not sorted!'
    assert len(spine1) > 0, 'spine1 is empty'
    assert len(spine1) == len(spine2), 'spines of different lengths'
    if points == 'all': points = range(len(spine1))
    displacement = displacement_along_curve(spine1, spine2, perpendicular=perpendicular, 
                                                    points=points)
    speed = displacement / len(points) / dt
    return speed

# TODO: rewrite using numpy
def displacement_along_curve(curve1, curve2, perpendicular=False, points='all'):
    """
        curve1 and curve2 are lists of tuples (x,y)
        the function return the sum of the 
        projections of (curve2-curve1) along curve1
        if perpendicular, the displacement is computed
        perpendicular to the spine
    """
    if points == 'all': points = range(len(curve1))
    total_displacement = 0.
    for i in points:
        if i + 1 < len(curve1):
            p2 = curve1[i + 1]
            p1 = curve1[i]
            p1_later = curve2[i]
            # positive number if p1_later gets closer to p2
            dist = get_direction_and_distance(p1, p2,
                                              p1, p1_later,
                                              perpendicular=perpendicular)
            total_displacement += dist
    return total_displacement

def compute_length(blob_id, **kwargs):
    times, spines, _ = get_data(blob_id, data_type='spine', **kwargs)
    na_values = ['', -1, 'NA', 'NaN', None, []]
    ltimes, lengths = [], []
    for t, spine in izip(times, spines):        
        if len(spine) != 0 or spine not in na_values:
            ltimes.append(t)
            lengths.append(length_of_spine(spine))
    return times, lengths

def compute_speed_along(blob_id, **kwargs):
    times, spines, _ = get_data(blob_id, data_type='spine', **kwargs)
    stimes, speeds = [], []
    for i in range(len(times)):
        if i + 1 < len(spines) and len(spines[i]) > 0 and len(spines[i + 1]) > 0:
            speed = speed_along_spine(spines[i], spines[i + 1],
                                      times[i + 1], times[i])
            stimes.append(time[i])
            speeds.append(speed)
    return stimes, speeds

def compute_speed_perp(blob_id, **kwargs):
    times, spines, _ = get_data(blob_id, data_type='spine', **kwargs)
    stimes, speeds = [], []
    perpendicular = True
    for i in range(len(times)):
        if i + 1 < len(spines) and len(spines[i]) > 0 and len(spines[i + 1]) > 0:
            speed = speed_along_spine(spines[i], spines[i + 1],
                                      times[i + 1], times[i], perpendicular=True)
            stimes.append(t)
            speeds.append(speed)
    return stimes, speeds

def compute_curvature(blob_id, **kwargs):
    times, spines, _ = get_data(blob_id, data_type='spine', **kwargs)
    ctimes, curvatures = [], []
    for t, spine in izip(times, spines):
        if len(spine) > 0:
            curve = curvature_of_spine(spine_list[i], points, scaling_factor)
            ctimes.append(t)
            curvatures.append(curve)
    return ctimes, curvatures

def curvature_of_spine(spine, points='all'):
    def compute_curvature_from_three_points(p, q, t):
        """
        Solution from: http://mathworld.wolfram.com/SSSTheorem.html
        p,q and t are tuples (x,y)
        """
        a = euclidean(p, q)
        b = euclidean(t, q)
        c = euclidean(p, t)
        if a == 0 or b == 0 or c == 0:
            return 0.
        s = 0.5 * (a + b + c)
        # preventing round off error from breaking code
        if (s * (s - a) * (s - b) * (s - c)) < 0:
            K = 0.
        else:
            K = math.sqrt(s * (s - a) * (s - b) * (s - c))
        return K / (a * b * c) * 4.
    if points == 'all':
        points = range(len(spine))
    curvatures = []
    for i in points:
        if 1 < i < len(spine) - 1:
            curvature = compute_curvature_from_three_points(spine[i - 1], spine[i], spine[i + 1])
            curvatures.append(curvature)
    return np.mean(curvatures)

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
    pull_all_for_blob_id(blob_id=bID)
    dur1 = time.time() - start
    start = time.time()
    pull_all_for_blob_id2(blob_id=bID)
    dur2 = time.time() - start

    print '1', dur1
    print '2', dur2
    print dur1/dur2
