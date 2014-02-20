#!/usr/bin/env python
'''
Author: Peter
Description: tests if google_spreadsheet_interface behaving properly
'''
# standard imports
import sys
import os
import unittest
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

# nonstandard imports
from measurement_switchboard import *

class TestMeasureSwitchboard(unittest.TestCase):
                    
    def test_measure_matches_metric_with_unicode_metric(self):
        metrics = SWITCHES.keys()
        N = len(metrics)
        for i in range(N):
            for j in range(N):
                measure = metrics[i]
                metric = unicode(metrics[j] + '_bl')
                if i == j:                    
                    self.assertTrue(measure_matches_metric(measure, metric))
                else:
                    self.assertFalse(measure_matches_metric(measure, metric))
                         
    def test_measure_matches_metric_against_all_metrics(self):
        metrics = SWITCHES.keys()
        N = len(metrics)
        for i in range(N):
            for j in range(N):
                measure = metrics[i]
                metric = metrics[j] + '_mm'
                if i == j:                    
                    self.assertTrue(measure_matches_metric(measure, metric))
                else:
                    self.assertFalse(measure_matches_metric(measure, metric))                                    
    def test_switchboard_function_args(self):
        metric = 'length_mm'
        f_sol = compute_length
        kwargs_sol = {'units': 'mm'}
        f, kwargs = switchboard(metric)
        self.assertTrue(f == f_sol)
        self.assertEqual(kwargs, kwargs_sol)
    
    def test_all_switchboard_function_args(self):
        for metric, options in SWITCHES.iteritems():
            # this copy is to prevent SWITCHES from being changed
            options = dict(options)
            f_sol = options.pop('func', None)
            permutations = [metric]
            solutions = [{}]
            for arg_type, opt_list in options.iteritems():
                #print arg_type, opt_list
                new, new_s = [], []
                for i in opt_list:
                    for s, p in zip(solutions, permutations):
                        newp = '{p}_{i}'.format(p=p,i=i)
                        #print '\t', s, p
                        news = {arg_type:i}
                        if s:
                            news.update(s)
                    new_s.append(news)
                    new.append(newp)                    
                permutations += new
                solutions += new_s
            for s, p in zip(solutions, permutations):
                f, kwargs = switchboard(p)
                if s != kwargs:
                    print p, s, kwargs
                self.assertEqual(f, f_sol)
                self.assertEqual(kwargs, s)

    def test_switchboard_function_args2(self):
        metric = 'curv_bl_mid'
        f_sol = compute_curvature
        kwargs_sol = {'units': 'bl', 'position':'mid'}
        f, kwargs = switchboard(metric)
        self.assertTrue(f == f_sol)
        self.assertEqual(kwargs, kwargs_sol)


    def test_if_all_SWITCHES_have_functions(self):
        for switch, options in SWITCHES.iteritems():
            func = options.get('func', None)
            self.assertTrue(func!=None)


if __name__ == '__main__':
    unittest.main()
        
    
