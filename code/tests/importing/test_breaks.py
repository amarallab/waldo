#!/usr/bin/env python
'''
Author: Peter
Description: generates two fake flag timedicts and makes sure breaks_and_coils successfully identifies 'good_regions'.
'''
# standard imports
import sys
import os
import unittest
import math
import random

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(HERE +'/../../importing'))
import flags_and_breaks

class TestPartManagement(unittest.TestCase):
    
    def test_find_no_breaks_for_sparse_flags(self):
        times = [(0.1 * i) for i in xrange(1000)]        
        flags = [True for _ in times]
        for i in [250, 500, 750]:
            flags[i] = False
        breaks = flags_and_breaks.create_break_list(times, flags)
        self.assertEqual(breaks, [])

    def test_find_all_bad_for_many_flags(self):
        times = [(0.1 * i) for i in xrange(1000)]        
        flags = [False for _ in times]
        for i in [250, 500, 750]:
            flags[i] = True         
        breaks = flags_and_breaks.create_break_list(times, flags)
        self.assertEqual(breaks, [(times[0], times[-1])])

    def test_find_breaks_bad_start(self):
        times = [(0.1 * i) for i in xrange(1000)]        
        switch,  flags = True, []
        for i, _ in enumerate(times):
            if i in [250, 500, 750]:
                switch = not switch
            flags.append(switch)
        breaks = flags_and_breaks.create_break_list(times, flags)
        break_solution = [(25.0, 49.9), (75.0, 99.9)]
        for b1, b2 in zip(breaks, break_solution):
            for i, j in zip(b1, b2):
                self.assertAlmostEqual(i, j)
               
    def test_find_breaks_bad_end(self):
        times = [(0.1 * i) for i in xrange(1000)]        
        switch,  flags = False, []
        for i, _ in enumerate(times):
            if i in [250, 500, 750]:
                switch = not switch
            flags.append(switch)
        breaks = flags_and_breaks.create_break_list(times, flags)
        break_solution = [(0.0, 24.9), (50.0, 74.9)]
        for b1, b2 in zip(breaks, break_solution):
            for i, j in zip(b1, b2):
                self.assertAlmostEqual(i, j)

    def report_mismatches(self, f1, f2):
        mismatches = []
        for i, (fi1, fi2) in enumerate(zip(f1,f2)):
            if fi1 != fi2:
                mismatches.append(i)
        print '{n} mismatches'.format(n= len(mismatches))
        print 'at positions {p}'.format(p=mismatches)            
        
    def test_flag_area(self):
        # function toggles
        min_ok_streak_len = 4
        N = min_ok_streak_len * 3               
        baseflags = [False for _ in xrange(N)]
        i = min_ok_streak_len
        for w in range(min_ok_streak_len*2):
            flags = list(baseflags)
            for j in xrange(i, i+w):
                flags[j] = True
            result_flags = flags_and_breaks.flag_trouble_areas(flags,
                                                               min_ok_streak_len)
            # if True streak is less than min window size, all Trues removed
            if w < min_ok_streak_len:
                if result_flags != baseflags:
                    self.report_mismatches(result_flags, baseflags)
                self.assertEqual(result_flags, baseflags)
            # if True streak greater than min window size, all Falses removed
            else:
                if result_flags != flags:
                    self.report_mismatches(result_flags, flags)
                self.assertEqual(result_flags, flags)                            
            filled_area = list(flags)                
    
    def test_remove_loner_flags(self):
        # function toggles
        N, window_size, min_nonflag_fraction = 20, 5 , 0.5
        function = flags_and_breaks.remove_loner_flags        
        baseflags = [True for _ in xrange(N)]
        w = window_size
        # change number of False flags in row
        for test_int in xrange(window_size):
            # test ability to find false flags along whole list
            flags = list(baseflags)
            i = N/2
            # add in test_int false flags in middle of flags
            for j in range(i, test_int+i):
                flags[j] = False                
            result_flags = function(flags, window_size, min_nonflag_fraction)
            # if number false flags less than min_nonflag_fraction, remove all
            if test_int <= (window_size * min_nonflag_fraction):
                if result_flags != baseflags:
                    print 'removal error i={i} testint={t}'.format(i=i, t=test_int)
                    print test_int, 'test'
                    print flags
                    self.report_mismatches(result_flags, baseflags)
                self.assertEqual(result_flags, baseflags)
            # if number false flags more than min_nonflag_fraction, keep all
            else:
                if result_flags != flags:
                    print 'leaving error i={i} testint={t}'.format(i=i, t=test_int)
                    print test_int, 'test'
                    print flags
                    self.report_mismatches(result_flags, flags)
                self.assertEqual(result_flags, flags)
            
if __name__ == '__main__':
    unittest.main()
