'''
Author: Peter Winter + Andrea L.
Date: jan 11, 2013
Description:
Pulls position timeseries out of the database and adds speed timeseries back into the database.
'''
# standard imports
import os
import sys
import math
from itertools import izip

# set paths
code_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)

# nonstandard imports
from database.mongo_retrieve import pull_data_type_for_blob
from database.mongo_retrieve import timedict_to_list
from filtering.equally_space import equally_space
from filtering.filter_utilities import compute_filtered_timedict
from filtering.filter_utilities import filter_stat_timedict as fst
from GeometricCalculations.shape_utilities import compute_angles4
from GeometricCalculations.shape_utilities import shapes_to_angles
from show_measure import quickplot_stat2

def get_string_from_float(t):
    return str(round(t, 3)).replace('.', '?')

def calculate_speeds(t, xy, scaling_factor=1):
    '''
    calculates instantanious velocity at each timepoint.
    :param t: list of times (floats)
    :param xy: list of (x,y) tuples indicating positions. x and y are floats.
    :param scaling_factor: divide distance at each timestep by this number to rescale speed.
    '''
    assert t.__eq__(sorted(t)), 'the times are not sorted!'
    time_speed_dict = {}
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        xy_future = xy[i + 1]
        xy_now = xy[i]
        ds = math.sqrt((xy_future[0] - xy_now[0]) ** 2 + (xy_future[1] - xy_now[1]) ** 2)
        ds = ds / scaling_factor
        time_speed_dict[get_string_from_float(t[i])] = ds / dt
    return time_speed_dict


def compute_centroid_measures(blob_id, steps_per_mm=50, metric='all', smooth=True, source_data_entry=None, **kwargs):
    ''' for blob_id pulls filtered xy data out of the database,
        calculates one or all metrics
    '''
    assert metric in ['all', 'centroid_speed_bl', 'centroid_speed']
    # pull source data from database

    datatype = 'xy_raw'
    if source_data_entry:
        assert source_data_entry['data_type'] == datatype, 'Error: wrong type of data entry provided'
    if not source_data_entry:
        source_data_entry = pull_data_type_for_blob(blob_id, data_type=datatype, **kwargs)
    xy_raw_timedict = source_data_entry['data']
    # smooth the raw xy coordinates
    xy_filtered_timedict = compute_filtered_timedict(xy_raw_timedict)
    # convert timedict into two sorted lists of times and positions
    t, xy = timedict_to_list(xy_filtered_timedict)

    # get the scaling factors
    # making robust to depreciated notation that should no longer be in database
    pixels_per_body_length = float(source_data_entry.get('midline-median', 1.0))
    if pixels_per_mm == 1.0:
        pixels_per_mm = float(source_data_entry.get('pixels_per_mm', 1.0))
    pixels_per_mm = float(source_data_entry.get('pixels-per-mm', 1.0))
    if pixels_per_body_length == 1.0:
        pixels_per_body_length = float(source_data_entry.get('pixels_per_body_length', 1.0))
        pixels_per_body_length = float(source_data_entry.get('midline_median', 1.0))

    if pixels_per_mm == 1: print blob_id, 'has no specified pixels_per_mm'
    if pixels_per_body_length == 1: print blob_id, 'has no specified pixels_per_body_length'

    elif metric == 'all':
        return {'centroid_speed_bl': calculate_speeds(t, xy, scaling_factor=pixels_per_body_length),
                'centroid_speed': calculate_speeds(t, xy, scaling_factor=pixels_per_mm)}
    elif metric == 'centroid_speed_bl':
        return calculate_speeds(t, xy, scaling_factor=pixels_per_body_length)
    elif metric == 'centroid_speed':
        return calculate_speeds(t, xy, scaling_factor=pixels_per_mm)

if __name__ == "__main__":

    blob_id = '20121117_175351_00133'
    #blob_id = '20121119_162934_07337'
    blob_id = '20121118_165046_01818'
    blob_id = '20130324_115435_04452'
    blob_id = '00000000_000001_00002'
    metric = 'centroid_ang_ds'
    metric = 'centroid_speed'
    metric = 'centroid_speed_bl'
    stat_timedict1 = compute_centroid_measures(blob_id, metric=metric, smooth=False)
    stat_timedict2 = compute_centroid_measures(blob_id, metric=metric, smooth=True)
    print len(stat_timedict1), len(stat_timedict2)
    quickplot_stat2(stat_timedict1, stat_timedict2, 'raw', 'smoothed')

    
