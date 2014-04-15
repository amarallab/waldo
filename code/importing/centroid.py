
'''
Date:
Description:
'''

__author__ = 'peterwinter'

# standard imports
import os
import sys
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../shared/')
PROJECT_DIR = os.path.abspath(HERE + '/../../')
EXCEPTION_DIR = PROJECT_DIR + '/data/importing/exceptions/'
sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR)

# nonstandard imports
from Encoding.decode_outline import pull_smoothed_outline, decode_outline
from ExceptionHandling.record_exceptions import write_pathological_input
from shared.wio.file_manager import get_timeseries, write_timeseries_file
from settings.local import SMOOTHING 
from equally_space import equally_spaced_tenth_second_times
from equally_space import equally_space_xy_for_stepsize
from filtering.filter_utilities import savitzky_golay, neighbor_calculation, domain_creator

from wormmetrics.compute_metrics import txy_to_speeds, angle_change_for_xy

ORDER = SMOOTHING['time_order']
WINDOW = SMOOTHING['time_window']

def process_centroid_old(blob_id, window=WINDOW, order=ORDER, store_tmp=True, **kwargs):
    """
    """
    # retrieve raw xy positions
    orig_times, xy = get_timeseries(blob_id, data_type='xy_raw', **kwargs)
    x, y = map(np.array, zip(*xy))
    # smooth across time
    x1 = savitzky_golay(x, window_size=window, order=order)
    y1 = savitzky_golay(y, window_size=window, order=order)
    # interpolate positions for equally spaced timepoints
    kind = 'linear'
    eq_times = equally_spaced_tenth_second_times(orig_times[0], orig_times[-1])
    #print orig_times[0], orig_times[-1]
    #print eq_times[0], eq_times[-1]
    interp_x = interpolate.interp1d(orig_times, x1, kind=kind)
    interp_y = interpolate.interp1d(orig_times, y1, kind=kind)        
    x_new = interp_x(eq_times)
    y_new = interp_y(eq_times)
    xy = zip(list(x_new), list(y_new))
    # store as cached file
    if store_tmp:
        #data ={'time':eq_times, 'data':xy}
        write_timeseries_file(ID=blob_id, data_type='xy',
                              times=eq_times, data=xy)
    return eq_times, xy

def process_centroid(blob_id, **kwargs):
    """
    """
    # retrieve raw xy positions
    orig_times, xy = get_timeseries(blob_id, data_type='xy_raw', **kwargs)
    x, y = zip(*xy)

    if xy == None:
        return 
    if len(xy) == 0:
        return 

    dataframe = full_package(orig_times, x, y)
    #print dataframe.head()

    times = list(dataframe.index)
    # write xy
    xy = zip(list(dataframe['x']), list(dataframe['y']))
    write_timeseries_file(ID=blob_id, data_type='xy',
                          times=times, data=xy)

    # write cent_speed
    cent_speed = dataframe['v']
    write_timeseries_file(ID=blob_id, data_type='cent_speed',
                          times=times, data=xy)

    # write angle_change
    angle_change = dataframe['dtheta']
    write_timeseries_file(ID=blob_id, data_type='angle_change',
                          times=times, data=xy)
    return

def full_package(times, x, y,
                 pre_smooth=(25, 5),
                 time_threshold = 30,
                 distance_threshold = 0.25,
                 max_score=500,
                 prime_smooth=(75, 5)):
    
    # perform preliminary smoothing on the xy data.
    window, order = pre_smooth
    x = savitzky_golay(y=np.array(x), window_size=window, order=order)
    y = savitzky_golay(y=np.array(y), window_size=window, order=order)
    
    # calculate stationary regions
    point_scores = neighbor_calculation(distance_threshold, x, y, max_score)
    domains = domain_creator(point_scores, timepoint_threshold=time_threshold)
    # make sure all x,y values stay consistant throughout stationary domains
    for (start, end) in domains:
        end += 1 #because string indicies are non inclusive
        l = end - start
        x[start:end] = np.ones(l) * np.median(x[start:end])
        y[start:end] = np.ones(l) * np.median(y[start:end])

    # perform primary smoothing on the xy data.
    window, order = prime_smooth
    x = savitzky_golay(y=np.array(x), window_size=window, order=order)
    y = savitzky_golay(y=np.array(y), window_size=window, order=order)

    eq_times = times
    # if setting spacing of timepoints, change this

    kind = 'linear'
    eq_times = equally_spaced_tenth_second_times(times[0], times[-1])
    #print times[0], times[-1]
    #print eq_times[0], eq_times[-1]
    interp_x = interpolate.interp1d(times, x, kind=kind)
    interp_y = interpolate.interp1d(times, y, kind=kind)        
    x = interp_x(eq_times)
    y = interp_y(eq_times)

    dataframe = xy_to_full_dataframe(times=eq_times, x=x, y=y)
    return dataframe

def xy_to_full_dataframe(times, x, y):    
    speeds = txy_to_speeds(t=times, x=x, y=y)
    angles = angle_change_for_xy(x, y, units='rad')
    print len(times), len(x), len(y), len(speeds), len(angles)
    df = pd.DataFrame(zip(x, y, speeds, angles), index=times,
                      columns=['x', 'y', 'v', 'dtheta'])
    print df.head()
    return df

if __name__ == '__main__':
    bi = '00000000_000001_00001'
    bi = '20130610_161943_20653'
    process_centroid(blob_id=bi)
