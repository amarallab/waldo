#!/usr/bin/env python

'''
Filename: show_measure
Description: a quick way to plot and look at values for a particular measure. 
particularly useful for checking filtering options.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'


# standard imports

import os
import sys
import pylab as pl

# path definitions
code_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)

# nonstandard imports
from database.mongo_retrieve import timedict_to_list
from filtering.filter_utilities import filter_stat_timedict as fst
from database.mongo_retrieve import timedict_to_list

def quickplot_stat(stat_timedict):

    # toggles
    #stat_timedict = size_timedict
    #stat_timedict = width_mm_timedict
    t, s = timedict_to_list(stat_timedict, remove_skips=True)
    for t1, s1 in zip(t, s):
        print type(t1), type(s1)
    s_filtered = fst(stat_timedict, return_type=list)
        

    pl.figure()
    pl.plot(s, color='red', label='raw')
    pl.plot(s_filtered, color='blue', label='stat filetered')
    pl.legend()
    pl.show()

def quickplot_stat2(stat_timedict1, stat_timedict2, label1, label2):

    # toggles
    #stat_timedict = size_timedict
    #stat_timedict = width_mm_timedict
    pl.figure()
    if len(stat_timedict1) > 1:
        t1, s1 = timedict_to_list(stat_timedict1, remove_skips=True)
        pl.plot(s1, color='red', marker='.', label=label1)
    if len(stat_timedict2) > 1:    
        t2, s2 = timedict_to_list(stat_timedict2, remove_skips=True)    
        pl.plot(s2, color='blue', label=label2)
    pl.legend()
    pl.show()

