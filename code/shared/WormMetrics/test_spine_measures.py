#!/usr/bin/env python
'''
Author: Peter
Description: tests the calculations performed on spine data.
'''
# standard imports
import sys
import os
import unittest
import math
import random
from spine_measures import *
import matplotlib.pyplot as plt
import numpy as np

class TestSpineMetrics(unittest.TestCase):    
    def test_compute_length(self):
        spine = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0],
                 [0.0, 10.0], [0.0, 0.0]]
        self.assertEqual(compute_length(spine), 40.0)

    def test_curvature(self):
        pass
        r = 2
        t = np.linspace(0.0, 3.14, 100)
        x, y = r*np.cos(t), r*np.sin(t)
        #plt.plot(x,y)
        #plt.show()

    
if __name__ == '__main__':
    unittest.main()
