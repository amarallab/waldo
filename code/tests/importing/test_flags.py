#!/usr/bin/env python

'''
Filename: test_flags.py

Description: unit tests to test flag_timepoints.py
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import unittest
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(HERE +'/../../importing'))

from flags_and_breaks import *

class TestFlagTimepoints(unittest.TestCase):
    def test_consolidate_all_true_flags(self):
        all_flags = {'f1': [True, True, True],
                     'f2': [True, True, True],}
        flags = consolidate_flags(all_flags)
        soln = [True, True, True]
        self.assertEqual(flags, soln)
        
    def test_consolidate_all_False_flags(self):
        all_flags = {'f1': [False, False],
                     'f2': [False, False],}
        flags = consolidate_flags(all_flags)
        soln = [False, False]
        self.assertEqual(flags, soln)

    def test_consolidate_mixed_flags(self):
        all_flags = {'f1': [False, True, True, True],
                     'f2': [False, True, True, True],
                     'f2': [True, True, False, True]}  
        flags = consolidate_flags(all_flags)
        soln = [False, True, False, True]
        self.assertEqual(flags, soln)
        
    def test_flag_no_outliers(self):
        fail_num = 0
        for i in range(10):
            normal_dataset = np.random.normal(loc=5.0, scale=1.0, size=1000)
            flags = flag_outliers(normal_dataset)
            soln = [True for i in range(1000)]
            if flags != soln:
                fail_num += 1
        self.assertTrue(fail_num <= 2)
        
    def test_flag_lage_and_small_outliers(self):
        N_normal, N_out1, N_out2 = 200, 50, 20
        normal_dataset = np.random.normal(loc=10.0, scale=1.0, size=N_normal)
        outliers1 = np.random.normal(loc=20.0, scale=1.0, size=N_out1)
        outliers2 = np.random.normal(loc=1.0, scale=0.5, size=N_out2)
        test_set = list(normal_dataset) + list(outliers1) + list(outliers2)
        flags = flag_outliers(test_set)
        soln = ([True] * N_normal) + ([False] * (N_out1 + N_out2))
        self.assertEqual(flags, soln)

    def test_flag_just_small_outliers(self):
        N_normal, N_out1, N_out2 = 200, 10, 30
        normal_dataset = np.random.normal(loc=10.0, scale=1.0, size=N_normal)
        outliers1 = np.random.normal(loc=20.0, scale=1.0, size=N_out1)
        outliers2 = np.random.normal(loc=1.0, scale=0.5, size=N_out2)
        test_set = list(normal_dataset) + list(outliers1) + list(outliers2)
        flags = flag_outliers(test_set, options='short')
        soln = ([True] * (N_normal+N_out1)) + ([False] * N_out2)
        self.assertEqual(flags, soln)

    def test_flag_just_large_outliers(self):
        N_normal, N_out1, N_out2 = 200, 10, 20
        normal_dataset = np.random.normal(loc=10.0, scale=1.0, size=N_normal)
        outliers1 = np.random.normal(loc=20.0, scale=1.0, size=N_out1)
        outliers2 = np.random.normal(loc=1.0, scale=0.5, size=N_out2)
        test_set = list(normal_dataset) + list(outliers1) + list(outliers2)
        flags = flag_outliers(test_set, options='long')
        soln = [True] * N_normal + [False] * N_out1 + [True] * N_out2
        self.assertEqual(flags, soln)

    def test_flag_outliers_with_nulls(self):        
        N_normal = 1000
        normal_dataset = np.random.normal(loc=10.0, scale=1.0, size=N_normal)
        nulls = [-1, '', [], 'NA', 'NaN']
        test_set = list(normal_dataset) + nulls 
        flags = flag_outliers(test_set)
        soln = [True] * N_normal + [False] * len(nulls)
        self.assertEqual(flags, soln)
                        
        
        
if __name__ == '__main__':
    unittest.main()
    
