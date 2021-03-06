3#!/usr/bin/env python

'''
Filename: test_blob_reader.py

Description: unit tests to test blob_reader.py
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import unittest
import math
import numpy as np
import random
#from angle_calculations import angle_change_for_xy2 as angle_change_for_xy
from angle_calculations import *

class TestAngleCalculations(unittest.TestCase):

    def test_angle_change_xy_for_clockwise_square(self):        
        xy = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
              (-1, 0), (-1, 1), (0, 1)]
        x, y = zip(*xy)
        soln = [-90.0, 0.0, -90.0, 0.0, -90.0, 0.0, -90.0]
        #print 'clockwise square'
        #print 'soln', soln
        #print angle_change_for_xy(x ,y)        
        self.assertAlmostEqual(soln, angle_change_for_xy(x ,y))
        
    def test_angle_change_xy_for_counter_clockwise_square(self):
        xy = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
              (1, 0), (1, 1), (0, 1)]
        x, y = zip(*xy)
        soln = [90.0, 0.0, 90.0, 0.0, 90.0, 0.0, 90.0]
        #print 'counter clockwise square'            
        #print 'soln', soln
        #print angle_change_for_xy(x ,y)
        self.assertAlmostEqual(soln, angle_change_for_xy(x ,y))        
        
    def test_angle_change_xy_for_back_and_forth(self):
        xy = [(0, 0), (1,0), (0,0), (-1, 0), (0,0)] 
        x, y = zip(*xy)
        soln = [180, 0, 180]
        #print 'back and forth'                        
        #print 'soln', soln        
        for i, j in zip(soln, angle_change_for_xy(x ,y)):
            self.assertAlmostEqual(i, j)

    def test_angle_change_xy_for_up_and_down(self):
        yx = [(0, 0), (1,0), (0,0), (-1, 0), (0,0)] 
        y, x = zip(*yx)
        soln = [180, 0, 180]
        #print 'up and down'                        
        #print 'soln', soln        
        for i, j in zip(soln, angle_change_for_xy(x ,y)):
            self.assertAlmostEqual(i, j)            

    def test_angle_change_for_clockwise_circle(self):
        r = 5
        t = np.linspace(3*np.pi, 0, 1000)
        y = r * np.sin(t)
        x = r * np.cos(t)
        # why is this the solution?
        soln = [-0.540540540] * (len(x) -1)
        for i, j in zip(soln, angle_change_for_xy(x ,y)):
            self.assertAlmostEqual(i, j)
                        
    def test_angle_change_for_counter_clockwise_circle(self):
        r = 1
        t = np.linspace(0, 3*np.pi, 1000)
        y = r * np.sin(t)
        x = r * np.cos(t)
        # why is this the solution?
        soln = [0.540540540] * (len(x) -1)
        for i, j in zip(soln, angle_change_for_xy(x ,y)):
            self.assertAlmostEqual(i, j)        

    def test_constant_anglar_change(self):

        N = 100
        x = [0]
        y = [0]
        v = 1.0
        theta = [0.0]
        d_theta = random.uniform(-1, 1)       
        for t in range(N):
            dx = v * np.cos(theta[-1])
            dy = v * np.sin(theta[-1])
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
            #Slightly change the direction based on the angle change.
            theta.append(theta[-1] + float(d_theta))
        for dt in angle_change_for_xy(x, y, units='rad'):
            self.assertAlmostEqual(d_theta, dt)
        
            
if __name__ == '__main__':
    unittest.main()
