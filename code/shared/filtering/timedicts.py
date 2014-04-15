#!/usr/bin/env python
'''
Filename: timedicts
Description: filtering functions to be preformed on data organized in timedicts.
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
 from filter_utilities import savitzky_golay

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
