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

def calculate_angular_speed(t, xy):
    '''
        the function computes all the angles
        between consecutive timepoints a time t[i], t[i+1]
        it then computes the angular change rate
        from the angle at time t[i], t[i+1] 
        to the one at time t[i+1], t[i+2]
        dt=t[i+2]-t[i] 
        delta_angle / dt 
        and delta_angle / ds
        where ds is computed from t[i] to t[i+2]
        times are just t[i]
    '''

    assert t.__eq__(sorted(t)), 'the times are not sorted!'
    time_ang_speed_dict = {}
    #time_overds_speed_dict = {}    
    # -2 because angles is one element shorter than t
    for i in range(len(t) - 2):
        dt = t[i + 2] - t[i]
        # three positions are required
        xy_now = xy[i]
        xy_future = xy[i + 1]
        xy_ffuture = xy[i + 2]

        # angle between i and i+1
        angle1 = compute_angles4(xy_now[0], xy_now[1],
                                 xy_future[0], xy_future[1])
        # angle between i+1 and i+2
        angle2 = compute_angles4(xy_future[0], xy_future[1],
                                 xy_ffuture[0], xy_ffuture[1])

        dangle = angle2 - angle1
        #ds=math.sqrt((xy_ffuture[0] - xy_now[0])**2 \
        #             +(xy_ffuture[1] - xy_now[1])**2 )
        #assert ds>0, 'the centroid did not move. Really?'

        # for the time we take the first point
        time_ang_speed_dict[get_string_from_float(t[i])] = dangle / dt
        #time_overds_speed_dict[get_string_from_float(t[i])]=dangle/ds

    return time_ang_speed_dict


def calculate_angular_speed_ds(xy, ds=1):
    '''
        the function computes all the angles
        between consecutive distance steps. 
    '''


    # equally spacing and computing angles
    equally_spaced_xy = equally_space(list(xy), prefixed_step=ds)
    #for i in range(len(equally_spaced_xy)-1):
    #    ds = math.sqrt((equally_spaced_xy[i+1][0] - equally_spaced_xy[i][0])**2 +
    #                   (equally_spaced_xy[i+1][1] - equally_spaced_xy[i][1])**2)
    #    print i, equally_spaced_xy[i], ds, pdist([equally_spaced_xy[i+1], equally_spaced_xy[i]])   

    equally_spaced_xy_old_format = []
    for p in equally_spaced_xy:
        equally_spaced_xy_old_format.append(p[0])
        equally_spaced_xy_old_format.append(p[1])
    angles = shapes_to_angles([equally_spaced_xy_old_format], False)[0]

    index_angular_speed_dict = {}
    for i in range(len(angles) - 1):
        dangle = angles[i + 1] - angles[i]
        index_angular_speed_dict[str(i)] = dangle

    return index_angular_speed_dict

def calculate_vibration(t, xy, scaling_factor=1):
    dts, dxdts, dydts = [], [], []
    for (t1, t2, xy1, xy2) in izip(t[:-1], t[1:], xy[:-1], xy[1:]):
        dt = t2 - t1
        x1, y1 = xy1
        x2, y2 = xy2
        dxdts.append((x2-x1) * scaling_factor / dt)
        dydts.append((y2-y1) * scaling_factor / dt)
    return t[:-1], dxdts, dydts


def compute_centroid_measures(blob_id, steps_per_mm=50, metric='all', smooth=True, source_data_entry=None, **kwargs):
    ''' for blob_id pulls filtered xy data out of the database,
        calculates one or all metrics
    '''
    assert metric in ['all', 'centroid_ang_ds', 'centroid_speed_bl', 'centroid_speed', 'vibration']
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

    #for i in source_data_entry:
    #    print i
    #exit

    # get the scaling factors
    pixels_per_mm = float(source_data_entry.get('pixels-per-mm', 1.0))
    #pixels_per_body_length = float(source_data_entry.get('pixels-per-body-length', 1.0))
    pixels_per_body_length = float(source_data_entry.get('midline-median', 1.0))

    # making robust to depreciated notation that should no longer be in database
    if pixels_per_mm == 1.0:
        pixels_per_mm = float(source_data_entry.get('pixels_per_mm', 1.0))
    if pixels_per_body_length == 1.0:
        pixels_per_body_length = float(source_data_entry.get('pixels_per_body_length', 1.0))
        pixels_per_body_length = float(source_data_entry.get('midline_median', 1.0))

    '''
    pixels_per_mm, pixels_per_body_length = 1.0, 1.0
    for k in source_data_entry.keys():
        if k.strip() == 'pixels_per_mm':
            pixels_per_mm = float(source_data_entry[k])
        if k.strip() == 'midline_median':
            pixels_per_body_length = float(source_data_entry[k])
    '''
    if pixels_per_mm == 1: print blob_id, 'has no specified pixels_per_mm'
    if pixels_per_body_length == 1: print blob_id, 'has no specified pixels_per_body_length'

    # computing time dicts
    if metric == 'all' and smooth:
        return {'centroid_ang_ds': fst(calculate_angular_speed_ds(xy, pixels_per_mm / steps_per_mm)),
                'centroid_speed_bl': fst(calculate_speeds(t, xy, scaling_factor=pixels_per_body_length)),
                'centroid_speed': fst(calculate_speeds(t, xy, scaling_factor=pixels_per_mm))}
    elif metric == 'centroid_ang_ds' and smooth:
        return fst(calculate_angular_speed_ds(xy, pixels_per_mm / steps_per_mm))
    elif metric == 'centroid_speed_bl' and smooth:
        return fst(calculate_speeds(t, xy, scaling_factor=pixels_per_body_length))
    elif metric == 'centroid_speed' and smooth:
        return fst(calculate_speeds(t, xy, scaling_factor=pixels_per_mm))
    elif metric == 'vibration':
        return calculate_vibration(t, xy, scaling_factor=pixels_per_mm)
    elif metric == 'all':
        return {'centroid_ang_ds': calculate_angular_speed_ds(xy, pixels_per_mm / steps_per_mm),
                'centroid_speed_bl': calculate_speeds(t, xy, scaling_factor=pixels_per_body_length),
                'centroid_speed': calculate_speeds(t, xy, scaling_factor=pixels_per_mm)}
    elif metric == 'centroid_ang_ds':
        return calculate_angular_speed_ds(xy, pixels_per_mm / steps_per_mm)
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

    
