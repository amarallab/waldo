'''
Date:
Description:
'''

__author__ = 'peterwinter'

# standard imports
import os
import sys
import numpy as np
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
from encoding.decode_outline import decode_outline
from deviant.record_exceptions import write_pathological_input
from wio.file_manager import get_timeseries, write_timeseries_file
from settings.local import SMOOTHING 
from equally_space import equally_spaced_tenth_second_times
from filtering.filter_utilities import savitzky_golay, neighbor_calculation, domain_creator
from metrics.compute_metrics import txy_to_speeds, angle_change_for_xy

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

def process_centroid(blob_id, verbose=True, **kwargs):
    """
    """
    # retrieve raw xy positions
    orig_times, xy = get_timeseries(blob_id, data_type='xy_raw', **kwargs)
    x, y = zip(*xy)

    if xy == None:
        return 
    if len(xy) == 0:
        return 

    dataframe, domains = full_package(orig_times, x, y)
    if verbose:
        print 'centroid measurements: \n'
        print dataframe.head(), '\n'

    times = list(dataframe.index)
    # write xy
    xy = zip(list(dataframe['x']), list(dataframe['y']))
    write_timeseries_file(ID=blob_id, data_type='xy',
                          times=times, data=xy)

    # write cent_speed
    cent_speed = list(dataframe['v'])
    write_timeseries_file(ID=blob_id, data_type='cent_speed',
                          times=times, data=cent_speed)

    # write angle_change
    angle_change = list(dataframe['dtheta'])
    write_timeseries_file(ID=blob_id, data_type='angle_change',
                          times=times, data=angle_change)

    # write is_moving durations
    if isinstance(domains, pd.DataFrame):
        write_timeseries_file(ID=blob_id, data_type='is_moving',
                              times=np.array(domains.index, dtype=float), 
                              data=np.array(domains[domains.columns], dtype=float))

    return

def domains_to_dataframe(stop_domains, times):    
    first, last = times[0], times[-1]

    if len(stop_domains) > 0:

        last_d = [0, stop_domains[0][0]]
        new_d = stop_domains[0]

        full = [(last_d[0], last_d[1], 1),
                (new_d[0], new_d[1], 0)]

        for d in stop_domains[1:]:
            new_d, last_d = d, new_d
            full.extend([(last_d[1], new_d[0], 1),  # moving
                         (new_d[0], new_d[1], 0)]) # not moving

        full.extend([(last_d[1], len(times)-1, 1)])
    else:
        full = [(0, len(times)-1, 1)]

    # convert from indicies to times.
    data = []
    t = times
    for start, stop, move in full:
        #print int(start), int(stop), len(times)
        data.append((t[int(start)], t[int(stop)], move))

    #df = pd.DataFrame()
    df = pd.DataFrame(data=data, columns=['start', 'end', 'moving'])
    df['dur'] =  df['end'] - df['start']
    df = df.set_index('start')

    print 'modes of movement:\n'
    print df.head(), '\n'

    return df


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

    # equally space 
    eq_times = times
    kind = 'linear'
    eq_times = equally_spaced_tenth_second_times(times[0], times[-1])
    interp_x = interpolate.interp1d(times, x, kind=kind)
    interp_y = interpolate.interp1d(times, y, kind=kind)        
    x = interp_x(eq_times)
    y = interp_y(eq_times)

    # change to dataframe
    dataframe = xy_to_full_dataframe(times=eq_times, x=x, y=y)
    domain_df = domains_to_dataframe(domains, times=times)
    return dataframe, domain_df

def xy_to_full_dataframe(times, x, y):    
    speeds = txy_to_speeds(t=times, x=x, y=y)
    angles = angle_change_for_xy(x, y, units='rad')
    #print len(times), len(x), len(y), len(speeds), len(angles)
    df = pd.DataFrame(zip(x, y, speeds, angles), index=times,
                      columns=['x', 'y', 'v', 'dtheta'])
    return df

if __name__ == '__main__':
    bi = '00000000_000001_00001'
    bi = '20130610_161943_20653'
    #bi = '20130415_104153_00853'
    process_centroid(blob_id=bi)
