#!/usr/bin/env python

'''
Filename: 
Discription: 
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'
# standard imports
import os
import sys
import unittest
from itertools import izip

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
assert os.path.exists(project_directory), 'project directory not found'
sys.path.append(project_directory)

# nonstandard imports
from Shared.Code.Database import manage_breaks as breaks
from SpineProcessing.Code.breaks_and_coils import create_break_dict, good_segments_from_timedict, combine_breakdicts


class TestBreakManagement(unittest.TestCase):
    """
    """
    
    def setUp(self):
        """
        """
        # create a set of timekeys similar to actual data
        times_per_s = 20
        total_seconds = 200
        times = [float(i)/times_per_s for i in range(0,times_per_s*total_seconds)]
        keys = map(breaks.float_to_key, times)
        # create a fake 1D timedict storing floats.
        timedict = {}
        for k, v in izip(keys, times): timedict[k] = v
        self.timedict = timedict
        # create flag breakpoints
        break_startends = [(100,150)]
        self.flag_breaks = self.make_fake_breakpoints(timedict, break_startends, 'general')
        # create coil breakpoints
        break_startends = [(10,50), (120, 160)]
        self.coil_breaks = self.make_fake_breakpoints(timedict, break_startends, 'coil')
        complete_break_dict = {}
        # add data and breakpoints into same timedict
        self.timedict_with_breaks = {}
        for i in [self.timedict, self.coil_breaks, self.flag_breaks]:
            self.timedict_with_breaks.update(i)

    def make_fake_breakpoints(self, timedict, break_startends, break_type='general', times_per_s=20):
        '''
        returns a dictionary of fake breakpoints starting and ending at times
        specified as a list of tuples 'break_startends'.
        inputs:
        timedict - any data timedict,required for create_break_dict()
                   {timekey:boolean, ...}
        break_startends - list of tuples [(t_start, t_end), ...]
        '''
        assert type(break_startends) == list
        times, keys = zip(*breaks.timedict_to_times(timedict))
        values = [True for k in keys]        
        for start_index, end_index in break_startends:
            for i in range(start_index*times_per_s, end_index*times_per_s): values[i] = False
        breakdict = {}
        for k, v in izip(keys, values): breakdict[k] = v
        return create_break_dict(breakdict, break_type)
        
    def test_break_combine_and_seperate(self):
        ''' test if we can combine and seperate dictionaries of breaks.
        '''
        # setup origional breakdicts and combine them
        breakdict1 = self.coil_breaks.copy()
        breakdict2 = self.flag_breaks.copy()            
        combined_breaks = combine_breakdicts(breakdict1, breakdict2)
        # check if all original entries in combined one.
        self.assertTrue((len(breakdict1)+len(breakdict2))==len(combined_breaks))
        for i in breakdict1:
            self.assertEqual(breakdict1[i], combined_breaks[i])
        for i in breakdict2:
            self.assertEqual(breakdict2[i], combined_breaks[i])    
        # split entry see if 
        split_breaks = breaks.split_breakdicts(combined_breaks)
        for break_type in split_breaks:
            assert break_type in ['coil', 'general', 'flag']
            break_dict = split_breaks[break_type]
            if break_type == 'coil':
                for t in break_dict:
                    self.assertEqual(break_dict[t], breakdict1[t])
            if break_type == 'general':
                for t in break_dict:
                    self.assertEqual(break_dict[t], breakdict2[t])
        # recombine entry see if it is same as first combined entry
        recombined_dict= {}
        for break_dict in split_breaks.values():
            recombined_dict = combine_breakdicts(recombined_dict,
                                                        break_dict)
        for i in recombined_dict:
            self.assertEqual(recombined_dict[t], combined_breaks[t])

    def test_break_and_timeseries_combine(self):        
        '''
        test if we can combine and seperate break dicts and timeseries
        '''
        # combine and seperate all breakdicts.
        td_breaks = self.timedict_with_breaks
        td = self.timedict
        breakdict, td_removed = breaks.seperate_breaks_from_timedict(td_breaks)
        # make sure timepoints are the same.
        for i in sorted(td_removed):
            self.assertTrue(i in td)
            self.assertEqual(td_removed[i], td[i])
        # make sure breaks are the same.
        for i in sorted(breakdict):
            self.assertTrue((i in self.coil_breaks) or (i in self.flag_breaks))
            if i in self.coil_breaks:
                self.assertEqual(breakdict[i], self.coil_breaks[i])
            if i in self.flag_breaks:
                self.assertEqual(breakdict[i], self.flag_breaks[i])
                    
    def test_segment_picking(self):
        # TODO: add in a shuffeling of regions.
        td_breaks = self.timedict_with_breaks
        good_regions = good_segments_from_timedict(td_breaks)
        self.assertEqual(3, len(good_regions))

        timedict = self.timedict
        break_startends = [(0,20), (180,200)]
        flag_breaks = self.make_fake_breakpoints(timedict, break_startends, 'general')
        for b in flag_breaks: print b, flag_breaks[b]
        timedict_with_breaks = {}
        timedict_with_breaks.update(timedict)
        timedict_with_breaks.update(flag_breaks)
        good_regions = good_segments_from_timedict(timedict_with_breaks)
        for g in good_regions:
            print g[0], g[-1], len(g)
        self.assertEqual(1, len(good_regions))

          

if __name__ == '__main__':
    unittest.main()



