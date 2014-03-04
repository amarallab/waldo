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
from filtering.filter_utilities import savitzky_golay

ORDER = SMOOTHING['time_order']
WINDOW = SMOOTHING['time_window']

def process_centroid(blob_id, window=WINDOW, order=ORDER, store_tmp=True, **kwargs):
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

def angle_change_for_xy(x, y):
    vectors = zip(np.diff(x), np.diff(y))
    vectors = vectors / np.linalg.norm(vectors)
    angles = np.arccos(np.dot(vectors[1:], vectors[:-1]))
    for i, (angle, v1, v2) in enumerate(zip(angles, vectors[1:], vectors[:-1])):
        if np.isnan(angle):
            if (v1 == v2).all():
                angles[i] ==  0.0
            else:
                angles[i] ==  np.pi
    return angles

def angle_change_over_distance(blob_id, store_tmp=True, step=0.1, **kwargs):
    times, xy = get_timeseries(blob_id, data_type='xy', **kwargs)
    times = np.array(times)
    x, y = map(np.array, zip(*xy))
    t, x, y = equally_space_xy_for_stepsize(times, x, y, step=step)
    angles = angle_change_for_xy(x, y)        

# defunct.
def show_centroid_processing(blob_id):
    # optionally show differences between origional, smoothed, and interpoated xy
    fig = plt.figure()
    ax1 = plt.subplot(2,1,1)
    plt.plot(orig_times,x, label='orig')
    plt.plot(orig_times,x1, label='smooth')
    plt.plot(eq_times,x_new, label='interp')
    plt.legend()
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    plt.plot(orig_times,y)
    plt.plot(orig_times,y1)
    plt.plot(eq_times,y_new)    
    plt.show()



if __name__ == '__main__':
    bi = '00000000_000001_00001'
    process_centroid(blob_id=bi)
