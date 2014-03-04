#!/usr/bin/env python

'''
Filename: breaks_and_coils.py

The segments of a timecourse are denoted either as a 'good region' or as a 'break' (which contains untrustworthy data).
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import pylab as pl
from itertools import izip
import math
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import scipy

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../shared/')
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.file_manager import get_timeseries, write_timeseries_file, write_metadata_file

# Globals
NULL_FLAGS = [-1, [], '', 'NA', 'NaN', u'', u'NA', u'NaN', None]

def consolidate_flags(all_flags):
    '''
    takes a dictionary of flag dictionaries and consolidates them into one
    flag dictionary. if any timepoint is false, the consolidated entry will read as false.
    '''
    if isinstance(all_flags, dict):
        flag_lists = [all_flags[i] for i in all_flags]
        return [all(i) for i in izip(*flag_lists)]
    else:
        return [all(i) for i in all_flags] 

def fit_gaussian(x, num_bins=200):
    # some testdata has no variance whatsoever, this is escape clause
    if math.fabs(max(x) - min(x)) < 1e-5:
        print 'fit_gaussian exit'
        return max(x), 1

    n, bin_edges = np.histogram(x, num_bins, normed=True)
    bincenters = [0.5 * (bin_edges[i + 1] + bin_edges[i]) for i in xrange(len(n))]

    # Target function
    fitfunc = lambda p, x: mlab.normpdf(x, p[0], p[1])

    # Distance to the target function 
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    # Initial guess for the parameters
    mu = np.mean(x)
    sigma = np.std(x)
    p0 = [mu, sigma]
    p1, success = scipy.optimize.leastsq(errfunc, p0[:], args=(bincenters, n))
    # weirdly if success is an integer from 1 to 4, it worked.
    if success in [1,2,3,4]:
        mu, sigma = p1
        return mu, sigma
    else:
        return None


def plot_fit_hists(x, num_bins=200, xlabel=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # the histogram of the data_from_list
    n, bins, patches = ax.hist(x, num_bins, normed=1, facecolor='green', alpha=0.25)
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    mu, sigma = fit_gaussian(x)
    y = mlab.normpdf(bincenters, mu, sigma)
    l = ax.plot(bincenters, y, 'r--', linewidth=2, label='fit dist')
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_xlim([0, mu + 10 * sigma])
    plt.show()
    plt.clf()

def calculate_threshold(x, p=0.05, acceptable_error=0.05, verbose=False):


    # some testdata has no variance whatsoever, this is escape clause
    if math.fabs(max(x) - min(x)) < 1e-5:
        if verbose:
            print 'no variance, skipping flagging'
        return min(x), max(x)
    if len(x) < 5:
        print 'where is x?', len(x), x

    mu = np.mean(x)
    sigma = np.std(x)
    # this is to avoid cutting out any measurements if they are all within an acceptable_error range
    if sigma / mu <= acceptable_error:
        return min(x), max(x)

    N = len(x)
    mu, sigma = fit_gaussian(x) 
    a = 2 * ((1 - p) ** (1. / N)) - 1
    b = scipy.special.erfinv(a)
    xmax = b * sigma * math.sqrt(2) + mu
    xmin = - b * sigma * math.sqrt(2) + mu

    #xmin, xmax = min(x), max(x)
    if verbose:
        print 'N=', N
        print 'fit sigma', sigma, 'fit mu', mu, 'min, max', min(x), max(x)
        print 'raw sigma', sigma, 'raw mu', mu, 'min, max', min(x), max(x)
        #print 'a%f' %a
        #print 'erfinv', b
        print 'min threshold', xmin, 'max threshold', xmax

    return xmin, xmax

def flag_outliers(values, options='both', null_flags=NULL_FLAGS):
    '''
    if flag == True, that timepoint is good.
    if flag == False, something wrong with timepoint
    '''

    assert options in ['short', 'long', 'both', 'bypass']

    # note: we have passed -1 in certain parts of the data to indicate an error in processing.
    # here we remove all -1s to calculate threshold, 
    # later we automatically flag all times that had -1       
    
    data = [d for d in values if d not in null_flags]
    min_threshold, max_threshold = calculate_threshold(data, p=0.05)

    # choose appropriate criterion. returns True if k passes.
    if options == 'both':
        flag_criterion = lambda k: min_threshold <= k <= max_threshold
    elif options == 'short':
        flag_criterion = lambda k: min_threshold <= k
    elif options == 'long':
        flag_criterion = lambda k: k <= max_threshold
        
    flags = []
    for k in values:
        if k in null_flags:
            flags.append(False)
        else:
            flags.append(flag_criterion(k))
    return map(bool, flags)

def flag_blob_data(blob_id, data_type, options='both', show_plot=False, verbose=True, **kwargs):
    '''
    inputs:
    blob_id - 
    data_type - 
    show_plot -
    '''
    times, data = get_timeseries(ID=blob_id, data_type=data_type, **kwargs)
    # check to see if all data is correct type
    err_msg = lambda x: 'Error: {bi} {dt} has a {t} ' \
                        'type value in data'.format(bi=blob_id, dt=data_type,
                                                    t=type(x))
        
    #for d in data:
    #    if type(d) not in [int, float, np.float, np.int]:
    #        print d, type(d)
    def check(x):
        assert type(x) in [int, float, np.float64, np.int], err_msg    
    map(check, data)
    # if more than half of points are zero, flag everything.
    N = len(data)
    zeros = len([x for x in data if x == 0.0])
    zeros += len([x for x in data if np.isnan(x)])
    if N ==0:
        return []
    if zeros / float(N) > 0.5:
        msg = '\t{dt} {p} % of N = {N}'
        msg += 'points are 0.0 | flag all'.format(p=round(100*zeros/float(N),
                                                          ndigits=2),
                                                          dt=data_type, N=N)
        print msg
        return [False] * N    
    data = [d for d in data if not np.isnan(d)]
    flags = flag_outliers(data, options=options)
    if show_plot:
        plot_fit_hists(data, xlabel=data_type)
    return flags

def flag_report(blob_id):
    def count_flagged_points(blob_id, data_type):
        x = flag_blob_data(blob_id, data_type, options='long', show_plot=False)
        l = len(x) - sum(x)
        x = flag_blob_data(blob_id, data_type, options='short', show_plot=False)
        s = len(x) - sum(x)
        print data_type, 'has %i flags (%i short and %i long) out of %i' % (l + s, s, l, len(x))
        #for i in sorted(x):
        #    print i, x[i]

    print 'for', blob_id
    count_flagged_points(blob_id, 'width20')
    count_flagged_points(blob_id, 'width50')
    count_flagged_points(blob_id, 'width80')
    count_flagged_points(blob_id, 'length_rough')
    #count_flagged_points(blob_id, 'spine_shift_speed')

def flag_blob_id(blob_id, verbose=True, store_tmp=True, **kwargs):
    times, _ = get_timeseries(blob_id, data_type='width20')
    '''
    all_flags = {'width20_flags': flag_blob_data(blob_id, 'width20', options='long', **kwargs),
                 'width50_flags': flag_blob_data(blob_id, 'width50', options='long', **kwargs),
                 'width80_flags': flag_blob_data(blob_id, 'width80', options='long', **kwargs),
                 'length_flags_short': flag_blob_data(blob_id, 'length_rough', options='short', 
                                                      **kwargs),
                 'length_flags_long': flag_blob_data(blob_id, 'length_rough', options='long', 
                                                     **kwargs),
                }
    '''

    flag_types = [('width20','long'), ('width50', 'long'), ('width80', 'long'),    
                  ('length_rough', 'short'), ('length_rough', 'long')]
                  
    N_rows = len(times)
    N_cols = len(flag_types)
    all_flags = np.zeros(shape=(N_rows, N_cols), dtype=bool)
    for i, (data_type, options)  in enumerate(flag_types):
        all_flags[:, i] = np.array(flag_blob_data(blob_id, data_type, options=options), dtype=bool)
    
    if store_tmp:
        write_timeseries_file(ID=blob_id, data_type='flags',
                              data=all_flags, times=times)
    if verbose:
        flags = consolidate_flags(all_flags)
        N = len(flags)
        # np.count_nonzero not in py2.6
        #good = np.count_nonzero(flag_list)
        good = len([i for i in flags if i])
        bad = N - good
        print '\tall flags: {flags} | N: {N} | {r} %'.format(flags=bad, N=N, r=100 * bad / N)
    return all_flags


def remove_loner_flags(flags, window_size=21, min_nonflag_fraction=0.7,
                       verbose=False):
    '''
    returns flags with scattered flags removed. Flags will be removed from resulting list if    
    the fraction of good timepoints within the running window size is greater than min_nonflag_fraction

    :param flags: the list of boolean values    
    :param window_size: the size of the window used to evaluate local (odd value int)
    :param min_nonflag_fraction: the fraction of flags around a loner_flag required to remove that loner_flag
    (float between 0 and 1)
    '''

    assert isinstance(window_size, int)
    assert 0.0 < min_nonflag_fraction < 1.0
    w = (window_size - 1) / 2
    new_flags = list(flags) # copy flags.
    for i, this_flag in enumerate(flags[w:-w], start=w):
        # if flag = false, test if loner, and if so, switch to true
        if this_flag == False:
            window = flags[i - w : i + w + 1]
            assert len(window) == window_size
            fraction_true = 1.0 * sum(window) / len(window)
            if fraction_true >= min_nonflag_fraction:
                new_flags[i] = True
            if verbose:
                print 'tested', i, this_flag, 'to', new_flags[i]
                print 'range:', window, fraction_true
    return new_flags


def plot_lists_of_flags(list_of_flags):
    """
    shows graphic representation of where flags occur in time course.

    :param list_of_flagdicts: a list containing several flag dicts.
    """
    numplots = len(list_of_flags)
    pl.figure()
    for i, flags in enumerate(list_of_flags):
        v = flags
        t = range(len(flags))
        pl.subplot(numplots, 1, i + 1)
        pl.plot(t, v)
        pl.fill_between(t, 1, v)
        pl.ylim([-0.1, 1.1])
    pl.show()

def create_breaks_for_blob_id(blob_id, verbose=True, store_tmp=True, **kwargs):
    """
    returns a dictionary of 'breaks' (ie. time segments for which we cannot trust the data)'

    :param blob_id: the blob_id
    :param verbose: toggle to turn on/off messages while running
    :param insert: toggle to turn on/off the import of resulting break_list into the database.
    """
    #times, all_flags = get_timeseries(blob_id, data_type='flags')
    times, all_flags = get_timeseries(ID=blob_id, data_type='flags')    
    # find general breakpoints
    flags = consolidate_flags(all_flags)
    break_list = create_break_list(times, flags)
    if verbose:
        print '\tbreaks from flags: {i}'.format(i=len(break_list))
    data_type = 'breaks'                            
    if store_tmp:
        write_metadata_file(blob_id, data_type=data_type, data=break_list)
    return break_list

def flag_trouble_areas(flags, min_ok_streak_len=10):
    '''
    returns a flags list in which flags any regions that are not 'ok' for longer than X consecutive time-points.
    (ie. trouble areas which alternate between being flagged/unflagged are flagged completely such that they will be
    discarded when breaks are calculated.)

    :param flags: the list of boolean values representing flags
    :param min_ok_streak_len: the minumum number of unflagged points
                              that must occur in a row for a region not to be flagged.
    '''
    new_flags = []
    all_ok_times, ok_streak = [], []

    for i, this_flag in enumerate(flags):
        if this_flag:
            ok_streak.append(i)
        else:
            if len(ok_streak) >= min_ok_streak_len:
                all_ok_times += ok_streak
            ok_streak = []
        # after loop ends, we need to check ok_streak one last time.
    if len(ok_streak) >= min_ok_streak_len:
        all_ok_times += ok_streak

    new_flags = [True if i in all_ok_times else False for i, _ in enumerate(flags)]
    return new_flags

def create_break_list(times, flags, verbose=False):
    """
    Returns a dictionary of 'breaks' which denote untrustworthy regions in the time-course.
    Note: best if flags has already been preprocessed using remove_loner_flags and then flag_trouble_areas

    :param flags: the flag dictionary used to create the break_list
    :param verbose: toggle to turn on/off messages while running
    :return:
    """
    flags_no_loners = remove_loner_flags(flags)
    flagged_areas = flag_trouble_areas(flags_no_loners)

    break_starts, break_ends = [], []    
    is_good, t = True, times[0]    
    for time, this_flag in izip(times, flagged_areas):
        is_good, was_good = this_flag, is_good
        t, last_t = time, t
        #print t, is_good
        if was_good and not is_good:
            break_starts.append(t)
            #print 'break start', t, is_good
        if is_good and not was_good:
            break_ends.append(last_t)
            #print 'break ends', t, is_good

    if len(break_starts) == len(break_ends) + 1:
        break_ends.append(times[-1])

    # match the start and endpoints together.
    msg = 'not an even number of starts and ends:'
    msg =  '{m} s{s} e{e}'.format(m=msg, s=len(break_starts), e=len(break_ends))
    #print len(break_ends), len(break_starts)
    assert len(break_ends) == len(break_starts), msg
    break_list = []
    # if no breaks found, return empty break list.
    if not break_starts:
        return []
    # otherwise continue
    for st, et in izip(break_starts, break_ends):
        # remove breaks where start and end points are equal
        if st != et:
            if verbose:
                print '\tbreak: {st} to {et}'.format(st=st, et=et)
            msg = 'start:{st}\tend:{et}\n'.format(st=st, et=et)
            assert st < et, msg
            break_list.append((st, et))
    return break_list

def get_flagged_times(blob_id):
    times, all_flags = get_timeseries(ID=blob_id, data_type='flags')    
    flags = consolidate_flags(all_flags)
    #flagged_times1 = []
    #for (t, f) in zip(times, flags):
    #    print t, f, f == False
    #    if f == False:
    #        flagged_times1.append(t)        
    #print 'are equal?', flagged_times == flagged_times1
    #print 'those were the flagged times!'
    return [t for (t, f) in zip(times, flags) if f==False]


def good_segments_from_data(break_list, times, data, flagged_times, verbose=True, null_flags=NULL_FLAGS):
    
    def remove_flagged_points(region, flagged_times, null_flags=null_flags):
        ''' returns a region with all the 
        flagged timepoints removed and all the timepoints with
        null data values removed. '''
        filtered_region = []
        for (t,d) in region:
            is_good = True
            for tf in flagged_times:
                if math.fabs(tf - t) < 0.05:
                    is_good = False
                # TODO: make this section a bit more robust.
                if type(d) == np.ndarray:
                    if d.any() in null_flags:
                        print d.shape
                        is_good = False
                elif d in null_flags:
                    is_good = False
            if is_good:
                filtered_region.append((t,d))
        return filtered_region
    
    if len(times) ==0:
        print 'error -- no times:', times
        return []

    if len(break_list) == 0:
        if verbose:
            s, e = times[0], times[-1]
            print '\tfrom start: {s}s | end: {e}s | N regions: 1'.format(s=s, e=e)
        # this returns a list with one region.
        return [remove_flagged_points(zip(times, data), flagged_times)]
    
    bstarts, bstops = zip(*break_list)
    #print bstarts, bstops

    # start these as false so 
    is_good, was_good = True, False
    region_start = times[0]
    good_regions, region, region_boundaries = [], [], []

    for t, datum in izip(times, data):        
        # for is_good to be true t cannot be between break start and stop 
        # and must have legit datum.
        new_bool = True            
        for start, stop in izip(bstarts, bstops):
            if (start <= t <= stop): # or not datum:
                new_bool = False

        # iterate is_good and was_good values, then decide what to do.
        (is_good, was_good) = (new_bool, is_good)
        if is_good and not was_good:
            # start a new region with this timepoin
            #print 'starting good region', t, len(datum) 
            region = [(t, datum)]
            region_start = t

        elif is_good and was_good:
            # keep extending the region if region has things in it.
            region.append((t, datum))
                
        elif not is_good and was_good and region:
            # region ended so add it to the good regions if anything is in region.
            #print 'ending good region', t, len(datum)
            good_regions.append(remove_flagged_points(region, flagged_times))
            region_boundaries.append((region_start, t))
        
    # if the loop ends with a good timepoint, store the region.
    if is_good:
        good_regions.append(remove_flagged_points(region, flagged_times))
        region_boundaries.append((region_start, t))
        
    if verbose:
        s, e, r = times[0], times[-1], len(region_boundaries)
        print '\tfrom start: {s}s | end: {e}s | N regions: {r}'.format(s=s, e=e, r=r)
        #print 'from', times[0], 'to', times[-1], 'regions are:', region_boundaries
    return good_regions

if __name__ == '__main__':
    #blob_id = '20120914_172813_01708'
    #print shift_index('3?000')

    blob_id = '00000000_000001_00003'
    blob_id = '20130319_150235_01070'
    blob_id = '20130320_164252_05955'
    blob_id = '00000000_000001_00008'
    blob_id = '20130319_150235_01830'
    blob_id = '20130719_124520_00951'
    print 'using blob_id', blob_id
    #find_coils(blob_id)

    flag_report(blob_id)
    flag_blob_id(blob_id)    
    
    breaks = create_breaks_for_blob_id(blob_id, insert=False)
    print breaks
    times, all_flags = get_timeseries(ID=blob_id, data_type='flags')
    flags = consolidate_flags(all_flags)
    flagged_times = get_flagged_times(blob_id)
    good_segments = good_segments_from_data(break_list=breaks, times=times, data=flags, 
                                            flagged_times=flagged_times)
    print type(good_segments)
    print len(good_segments)
    for i in good_segments:
        print
        print i
        print
    
