#!/usr/bin/env python
'''
Filename: filter_utilities
Description: functions involving smoothing of data.
'''

__authors__ = 'Peter B. Winter and Andrea Lancanetti'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import numpy as np
import scipy
import scipy.signal
from itertools import izip

# manage paths
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
shared_directory = project_directory + 'code/shared/'
assert os.path.exists(shared_directory), 'shared directory not found'
sys.path.append(shared_directory)

# nonstandard imports
#from PrincipalComponents.utilities import compute_transpose
from equally_space import equally_space_snapshots_in_time, compute_transpose
from database.mongo_retrieve import timedict_to_list

def smoothing_function(xyt, window, order):
    """
    Takes in a list of positions and times of the worm, the window size for the smoothing function, and the
    polynomial order for the smoothing function. Returns a list of the smoothed x and smoothed y values
    with unchanged time values.
    """
    from filter_utilities import savitzky_golay
    x, y, t = zip(*xyt)
    smooth_x = savitzky_golay(y=np.array(x), window_size=window, order=order)
    smooth_y = savitzky_golay(y=np.array(y), window_size=window, order=order)
    return zip(smooth_x, smooth_y, t)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
        the values of the time history of the signal.
        window_size : int
        the length of the window. Must be an odd integer number.
        order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
        deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The find_ex_ids_to_update idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
        """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in xrange(order + 1)] for k in xrange(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def smooth_snapshots(original_snapshots, filter_method='savitzky_golay', running_window_size=19, order=4):
    """
        The snapshots are expected to be equally spaced in time.

        INPUT:
        
        1. original_snapshots is a list of list, so that original_snapshots[i] looks like [x1, y1, x2, y2, ...]
        2. filter_method is a keyword, whoch could be 'wiener', 'median' or 'savitzky_golay'
        3. running_window_size : int
           the length of the window. Must be an odd integer number.
        4. order : int. Just for the savitzky_golay method
           the order of the polynomial used in the filtering.
           Must be less then `window_size` - 1.
           
        OUTPUT:
        
            filtered snapshots
    """

    tsnapshots = compute_transpose(original_snapshots)
    smoothed_snaps = []

    for tsnapshot in tsnapshots:

        if filter_method == 'median':
            filtered_signal = scipy.signal.medfilt(tsnapshot, kernel_size=running_window_size)

        if filter_method == 'wiener':
            filtered_signal = scipy.signal.wiener(tsnapshot)

        if filter_method == 'savitzky_golay':
            filtered_signal = savitzky_golay(np.array(tsnapshot), running_window_size, order)

        smoothed_snaps.append(filtered_signal)

    return compute_transpose(smoothed_snaps)


def compute_polynomial_xy_smoothing_by_index(unfiltered_outline, window_size=23, poly_order=3):
    '''
        unfiltered_outline is a list of tuples/lists: [[x1,y1], [x2,y2], ...]
        returns outline in the same format
    '''

    xs, ys = zip(*unfiltered_outline)

    xs = xs[-3:] + xs[:] + xs[:3]
    ys = ys[-3:] + ys[:] + ys[:3]

    filtered_xs = list(savitzky_golay(np.array(xs), window_size, poly_order))
    filtered_ys = list(savitzky_golay(np.array(ys), window_size, poly_order))

    filtered_xs = filtered_xs[3:-2]
    filtered_ys = filtered_ys[3:-2]

    filtered_outline = zip(filtered_xs, filtered_ys)

    return filtered_outline


def smooth_and_equally_space(ids, snapshots, running_window_size=19, order=4, filter_method='savitzky_golay'):
    """
        1. ids_strings is a list of strings with the times of the snapshots
        2. snapshots is a list of list, so that snapshots[i] looks like [x1, y1, x2, y2, ...]        
        2. filter_method is a keyword, whoch could be 'wiener', 'median' or 'savitzky_golay'
        3. running_window_size : int
        the length of the window. Must be an odd integer number.
        4. order : int. Just for the savitzky_golay method
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.

        returns equally-spaced times (strings), equally-spaced snapshots, and smooth snapshots
    """

    ids_eq, snapshots_eq = equally_space_snapshots_in_time(ids, snapshots)
    filtered_snapshots = smooth_snapshots(snapshots_eq, filter_method=filter_method,
                                          running_window_size=running_window_size, order=order)
    return ids_eq, snapshots_eq, filtered_snapshots

def smooth_and_equally_space_point_format(ids, points, running_window_size=19, order=4, filter_method='savitzky_golay'):
    values = []
    for ps in points:
        value = []
        for xy in ps:
            value.append(xy[0])
            value.append(xy[1])
            #print len(value)
        values.append(value)
        #print len(values)
    times, snapshots_eq, filtered_snapshots = smooth_and_equally_space(ids, values,
                                                                       running_window_size=running_window_size,
                                                                       order=order, filter_method=filter_method)

    filtered_points = []
    for v in filtered_snapshots:
        xs = v[::2]
        ys = v[1::2]
        xy = zip(xs, ys)
        filtered_points.append(xy)
    return times, filtered_points

def filter_time_series(times, values):
    '''
    inputs:
    times - list of floats
    Wrong! values - a list of lists (ex. [[x1,y1],[x2,y2]])

    outputs:
    filtered_snapshots - a list of lists (ex. [[
    '''
    ids_eq, snapshots_eq, filtered_snapshots = smooth_and_equally_space(times, values,
                                                                        running_window_size=19,
                                                                        order=4,
                                                                        filter_method='savitzky_golay')
    times = [float(i) for i in ids_eq]
    return times, filtered_snapshots


def compute_filtered_timedict(xy_raw_timedict):
    # convert timedict into two sorted lists of times and positions
    t, xy = timedict_to_list(xy_raw_timedict)
    # process lists of times and positions
    times, xy_filtered = filter_time_series(t, xy)
    # convert lists back into a timedict
    filtered_timedict = {}
    for time, xy_f in izip(times, xy_filtered):
        time_key = str('%.3f' % time).replace('.', '?')
        filtered_timedict[time_key] = xy_f
        # insert filtered xy positions back into the database
    return filtered_timedict


def filter_stat_timedict(stat_timedict, return_type=dict):
    from itertools import izip
    assert type(stat_timedict) == dict
    assert return_type in [list, dict]

    # TODO: do not remove skips but smooth each region seperatly
    times, stats = timedict_to_list(stat_timedict, remove_skips=False)

    stat_region = []
    region_times = []
    all_stats = []
    filtered_timedict = {}
    
    for i, (t, s) in enumerate(izip(times, stats), start=1):
        if s == 'skipped' or s == []:
            # if it's long enough, filter the region.
            if len(stat_region) >= 26:
                filtered_stat = savitzky_golay(np.array(stat_region), window_size=13, order=4)
                
                all_stats += list(stat_region)
                for tk, fs in izip(region_times, filtered_stat): filtered_timedict[tk] = fs
                  
            # if region, too short, don't filter, but leave data in.
            else: 
                #print len(stat_region), 'stat region'
                all_stats += stat_region
                for tk, fs in izip(region_times, stat_region): filtered_timedict[tk] = fs
                    
            # keep the skipped in the timedict.
            all_stats.append(s)
            filtered_timedict[('%.3f' % t).replace('.', '?')] = s
            # reset region.
            stat_region = []
            region_times = []

        # if this is the last point, checkout
        elif i >= len(times):
            stat_region.append(s)
            region_times.append(('%.3f' % t).replace('.', '?'))            
            if len(stat_region) >= 26:
                filtered_stat = savitzky_golay(np.array(stat_region), window_size=13, order=4)
            
                all_stats += list(filtered_stat)
                for tk, fs in izip(region_times, filtered_stat): filtered_timedict[tk] = fs
                  
            
            else: 
                #print len(stat_region), 'stat region'
                all_stats += stat_region
                for tk, fs in izip(region_times, stat_region): filtered_timedict[tk] = fs

        else:
            stat_region.append(s)
            region_times.append(('%.3f' % t).replace('.', '?'))

    
    if return_type == list:
        return all_stats
    else:
        return filtered_timedict

if __name__ == '__main__':
    """
        running this script shows a simple example
    """

    print 'testing filtering...'

    import random
    # EXAMPLE: define times with some noise
    times = list(np.linspace(0, 30, num=31))
    times = [t + random.uniform(-0.2, 0.2) for t in times]

    print times

    # EXAMPLE: define time series with some noise
    values = [np.sin(0.1 * t) + np.sin(10 * t) + random.uniform(-0.2, 0.2) for t in times]
    ids = [str(t) for t in times] # times could be string (or not)

    print ids
    print values

    ids_eq, snapshots_eq, filtered_snapshots = smooth_and_equally_space(ids, compute_transpose([values]),
                                                                        running_window_size=13, order=3,
                                                                        filter_method='median')

    snapshots_eq = compute_transpose(snapshots_eq)
    filtered_snapshots = compute_transpose(filtered_snapshots)

    print len(filtered_snapshots), len(filtered_snapshots[0])

    import matplotlib.pyplot as plt

    plt.plot(ids, values, 'o')
    plt.plot(ids_eq, snapshots_eq[0], 'x', ls='-')
    plt.plot(ids_eq, filtered_snapshots[0], 'x', ls='-')
    plt.legend(['data', 'eq', 'smoothed'], loc='upper right')
    plt.savefig('test1.pdf')
