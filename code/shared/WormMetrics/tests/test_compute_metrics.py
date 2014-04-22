#!/usr/bin/env python
'''
Author: Peter
Description: tests the calculations performed in compute_metrics
'''
# standard imports
import sys
import os
import unittest
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# nonstandard imports
from compute_metrics import *

    
class TestComputeMetrics(unittest.TestCase):    
    def test_length_of_spine(self):
        spine = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0],
                 [0.0, 10.0], [0.0, 0.0]]
        self.assertEqual(length_of_spine(spine), 40.0)

    def test_curvature_of_straight_spine(self):
        spine = [(i, 0.0) for i in np.arange(0.0, 10.0, 1)]
        self.assertAlmostEqual(0.0, curvature_of_spine(spine))
        
    def test_curvature_of_curcular_spine(self):
        r = 10.0
        t = np.linspace(0.0, 3.14, 50)
        x, y = r*np.cos(t), r*np.sin(t)
        spine = zip(x, y)
        self.assertAlmostEqual(1/r, curvature_of_spine(spine))

    def test_parallel_displacement_along_straight_curves(self):
        curve1 = [(i, 0.0) for i in np.arange(0.0, 10.0, 1)]
        curve2 = [(i, 0.0) for i in np.arange(0.1, 10.1, 1)]
        along = displacement_along_curve(curve1, curve2)
        perp = displacement_along_curve(curve1, curve2, perpendicular=True)
        #print 'move along'
        #print along, displacement_along_curve2(curve1, curve2)
        #print perp, displacement_along_curve2(curve1, curve2, perpendicular=True)
        self.assertAlmostEqual(-0.9, along)
        self.assertEqual(0.0, perp)

    def test_perp_displacement_along_straight_curves(self):
        curve1 = [(i, 0.0) for i in np.arange(0.0, 10.0, 1)]
        curve2 = [(i, 0.1) for i in np.arange(0.0, 10.0, 1)]                    
        along = displacement_along_curve(curve1, curve2)
        perp = displacement_along_curve(curve1, curve2, perpendicular=True)
        #print 'move perp'        
        #print along, displacement_along_curve2(curve1, curve2)
        #print perp, displacement_along_curve2(curve1, curve2, perpendicular=True)
        self.assertAlmostEqual(0.0, along)
        self.assertAlmostEqual(0.9, perp)

    def test_mixed_displacement_along_straight_curves(self):
        curve1 = [(i, 0.0) for i in np.arange(0.0, 10.0, 1)]
        curve2 = [(i, 0.1) for i in np.arange(0.1, 10.1, 1)]                    
        along = displacement_along_curve(curve1, curve2)
        perp = displacement_along_curve(curve1, curve2, perpendicular=True)
        #print 'move perp'        
        #print along, displacement_along_curve2(curve1, curve2)
        #print perp, displacement_along_curve2(curve1, curve2, perpendicular=True)
        self.assertAlmostEqual(-0.9, along)
        self.assertAlmostEqual(0.9, perp)
        
                
if __name__ == '__main__':
    unittest.main()

