#!/usr/bin/env python
'''
Filename: filter_utilities
Description: functions involving smoothing of data.
'''
from filtering.equally_space_old import equally_space_times, linear_interpolation

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

# these functions may be useful if additional domain manipulation is added.

# def expand_domains(xy, domains):
#     #""" expands the one value left for each domain into multiple values """
#     expansions = {}
#     shift = 0
#     for start, end in domains:
#         #expansions[start - shift] = end - start + 1
#         #shift += end - start
#         expansions[start - shift] = end - start
#         shift += end - start - 1
#
#     #print expansions
#     expanded = []
#     for i, val in enumerate(xy):
#         if i in expansions:
#             segment = [val] * expansions[i]
#         else:
#             segment = [val]
#         expanded.extend(segment)
#     return expanded

def reduce_domains(xy, domains):
    #""" removes all but one value for each domain """
    reduction_filter = [True] * len(xy)
    for start, end in domains:
        reduction_filter[start + 1: end] = [False] * (end - start - 1)
    return [v for (f,v) in zip(reduction_filter, xy) if f]


def neighbor_calculation(distance_threshold, x, y, max_score=500):
    """
    Calculates the score at each point in the worm path, where the score represents the number
    of neighbors within a certain threshold distance.
    :param distance_threshold: This is the threshold distance to be considered a neighbor to a point, assumed here
    to be in mm, not pixels
    :param nxy: A list of xy points, presumed to be noisy, i.e. the recorded data of the worm path
    :param max_score: the max number of timepoints that will be tested as neigbors.
    :return: A list of positive integer scores for each point in the worm path.
    """
    point_scores = []
    for i, pt in enumerate(izip(x,y)):
        start = max([0, i-max_score])
        if i < 3:
            point_scores.append(0)
            continue

        #short_list = xy[start:i]
        #xi, yi = zip(*short_list)
        xi, yi = x[start:i], y[start:i]
        dx = np.array(xi) - pt[0]
        dy = np.array(yi) - pt[1]
        dists = np.sqrt(dx**2 + dy**2)

        score = 0
        for d in dists[::-1]:
            if d < distance_threshold:
                score += 1
            else:
                break
        #print i, score, dists[0], dists[-1]
        point_scores.append(score)
    return point_scores

def domain_creator(point_scores, timepoint_threshold=30):
    """
    Calculates the domains, or regions along the worm path that exhibit zero movement. It does so by setting
    a threshold score for the number of nearby neighbors necessary to be considered an area of no movement, and
    then setting a domain behind that point to be considered non moving.

    :param point_scores: The neighbor scores of each point in the worm path
    :param timepoint_threshold: The score threshold for a point to be considered the initiator of a non-moving domain
    :return: A list of domains
    """
    #print 'Calculating Domains'
    threshold_indices = []
    for a, score in enumerate(point_scores):
        if score > timepoint_threshold:
            threshold_indices.append(a)
    domains = []
    for index in threshold_indices:
        score = point_scores[index]
        left = index-score
        if left < 0:
            left = 0
        if domains:
            if domains[-1][1] >= left:
                domains[-1] = [domains[-1][0], index]
            else:
                domains.append([left, index])
        else:
            domains.append([left, index])

    #Takes in the list of domains and merges domains that overlap. This function isn't totally
    #necessary anymore, since I've sort of implemented this logic into the initial calculation

    new_domains = []
    for a, domain in enumerate(domains[:-1]):
        if domain[1] > domains[a+1][0]:
            new_domains.append([domain[0], domains[a+1][1]])
    if not new_domains:
        new_domains = domains

    #print 'Number of domains: {d}'.format(d=len(domains))
    return new_domains


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

#
# def smooth_snapshots(original_snapshots, filter_method='savitzky_golay', running_window_size=19, order=4):
#     """
#         The snapshots are expected to be equally spaced in time.
#
#         INPUT:
#
#         1. original_snapshots is a list of list, so that original_snapshots[i] looks like [x1, y1, x2, y2, ...]
#         2. filter_method is a keyword, whoch could be 'wiener', 'median' or 'savitzky_golay'
#         3. running_window_size : int
#            the length of the window. Must be an odd integer number.
#         4. order : int. Just for the savitzky_golay method
#            the order of the polynomial used in the filtering.
#            Must be less then `window_size` - 1.
#
#         OUTPUT:
#
#             filtered snapshots
#     """
#
#     tsnapshots = compute_transpose(original_snapshots)
#     smoothed_snaps = []
#
#     for tsnapshot in tsnapshots:
#
#         if filter_method == 'median':
#             filtered_signal = scipy.signal.medfilt(tsnapshot, kernel_size=running_window_size)
#
#         if filter_method == 'wiener':
#             filtered_signal = scipy.signal.wiener(tsnapshot)
#
#         if filter_method == 'savitzky_golay':
#             filtered_signal = savitzky_golay(np.array(tsnapshot), running_window_size, order)
#
#         smoothed_snaps.append(filtered_signal)
#
#     return compute_transpose(smoothed_snaps)
#
#
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

#
# def smooth_and_equally_space(ids, snapshots, running_window_size=19, order=4, filter_method='savitzky_golay'):
#     """
#         1. ids_strings is a list of strings with the times of the snapshots
#         2. snapshots is a list of list, so that snapshots[i] looks like [x1, y1, x2, y2, ...]
#         2. filter_method is a keyword, whoch could be 'wiener', 'median' or 'savitzky_golay'
#         3. running_window_size : int
#         the length of the window. Must be an odd integer number.
#         4. order : int. Just for the savitzky_golay method
#         the order of the polynomial used in the filtering.
#         Must be less then `window_size` - 1.
#
#         returns equally-spaced times (strings), equally-spaced snapshots, and smooth snapshots
#     """
#
#     ids_eq, snapshots_eq = equally_space_snapshots_in_time(ids, snapshots)
#     filtered_snapshots = smooth_snapshots(snapshots_eq, filter_method=filter_method,
#                                           running_window_size=running_window_size, order=order)
#     return ids_eq, snapshots_eq, filtered_snapshots

# def smooth_and_equally_space_point_format(ids, points, running_window_size=19, order=4, filter_method='savitzky_golay'):
#     values = []
#     for ps in points:
#         value = []
#         for xy in ps:
#             value.append(xy[0])
#             value.append(xy[1])
#             #print len(value)
#         values.append(value)
#         #print len(values)
#     times, snapshots_eq, filtered_snapshots = smooth_and_equally_space(ids, values,
#                                                                        running_window_size=running_window_size,
#                                                                        order=order, filter_method=filter_method)
#
#     filtered_points = []
#     for v in filtered_snapshots:
#         xs = v[::2]
#         ys = v[1::2]
#         xy = zip(xs, ys)
#         filtered_points.append(xy)
#     return times, filtered_points


# def filter_time_series(times, values):
#     '''
#     inputs:
#     times - list of floats
#     Wrong! values - a list of lists (ex. [[x1,y1],[x2,y2]])
#
#     outputs:
#     filtered_snapshots - a list of lists (ex. [[
#     '''
#     ids_eq, snapshots_eq, filtered_snapshots = smooth_and_equally_space(times, values,
#                                                                         running_window_size=19,
#                                                                         order=4,
#                                                                         filter_method='savitzky_golay')
#     times = [float(i) for i in ids_eq]
#     return times, filtered_snapshots

# def compute_transpose(x):
#     """ given a matrix, computes the transpose """
#     xt=[([0]*len(x)) for k in x[0]]
#     for i, x_row in enumerate(x):
#         for j, b in enumerate(x_row):
#             xt[j][i]=x[i][j]
#     return xt


if __name__ == '__main__':
    print 'hello'

    #running this script shows a simple example


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

    # todo: finish example
    #
    # import matplotlib.pyplot as plt
    #
    # plt.plot(ids, values, 'o')
    # plt.plot(ids_eq, snapshots_eq[0], 'x', ls='-')
    # plt.plot(ids_eq, filtered_snapshots[0], 'x', ls='-')
    # plt.legend(['data', 'eq', 'smoothed'], loc='upper right')
    # plt.savefig('test1.pdf')


# def equally_space_snapshots_in_time(ids_strings, original_snapshots):
#     """
#         ids_strings is a list of strings with the times of the snapshots
#         original_snapshots is a list of list, so that original_snapshots[i] looks like [x1, y1, x2, y2, ...]
#         returns equally spaced times and snapshots
#     """
#
#     num_ids = [float(v) for v in ids_strings]
#     ids_eq = equally_space_times(num_ids)
#
#     tsnapshots = compute_transpose(original_snapshots)
#     equally_spaced_snapshots = []
#
#     for tsnapshot in tsnapshots:
#         eq_values = equally_space_snapshots_in_time_1d(num_ids, ids_eq, tsnapshot)
#         equally_spaced_snapshots.append(eq_values)
#
#     ids_strings_eq = [str(v) for v in ids_eq]
#
#     return ids_strings_eq, compute_transpose(equally_spaced_snapshots)
#
#
# def equally_space_snapshots_in_time_1d(ids_not_eq, ids_eq, values, interpolation_kind='linear'):
#     """
#         ids_not_eq and ids_eq are lists of floats with times
#         values is a list of floats you want to equally space in time
#         interpolation_kind should be 'linear'. nothing more for now.
#         returns equally spaced positions
#     """
#
#     assert ids_not_eq == sorted(ids_not_eq), 'DATA PROBLEM: time ids are not sorted!'
#     assert ids_eq == sorted(ids_eq), 'BUG!!! ids_eq not sorted!!!'
#
#     import bisect
#
#     eq_values = []
#     for k, dummy in enumerate(ids_eq):
#
#         left_index = bisect.bisect_left(ids_not_eq, ids_eq[k])
#         assert left_index < len(ids_not_eq), 'left index is out of range'
#
#         if ids_not_eq[left_index] == ids_eq[k]:
#             eq_values.append(values[left_index])
#
#         else:
#             left_index -= 1
#             right_index = left_index + 1
#             #print left_index, right_index, len(ids_eq), len(ids_not_eq), len(values)
#             eq_values.append(linear_interpolation( \
#                 ids_eq[k],
#                 (ids_not_eq[left_index], values[left_index]), \
#                 (ids_not_eq[right_index], values[right_index])) \
#                 )
#
#             #print ids_eq[k], ids_not_eq[left_index], ids_not_eq[right_index], 'III', left_index, right_index, k
#             #print eq_values[-1], values[left_index], values[right_index]
#             assert ids_eq[k] >= ids_not_eq[left_index] and ids_eq[k] < ids_not_eq[right_index], 'bisection is wrong'
#
#     return eq_values
