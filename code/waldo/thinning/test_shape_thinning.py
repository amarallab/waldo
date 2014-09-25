#!/usr/bin/env python

'''
Filename: test_blob_reader.py

Description: unit tests to test blob_reader.py
'''
from __future__ import absolute_import

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard library
import unittest

# third party
import numpy as np

# package specific
from .shape_thinning import skeletonize2 as skeletonize

class TestShapeThinning(unittest.TestCase):

    def compare_arrays(self, a1, a2):
        for r1, r2 in zip(a1, a2):
            for i1, i2 in zip(r1, r2):
                self.assertEqual(i1, i2)


    def test_thin_already_thinned_flat_shape(self):
        start = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0]])
        end = skeletonize(start)
        self.compare_arrays(start, end)

    def test_thin_already_thinned_long_shape(self):
        start = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 0]])
        end = skeletonize(start)
        self.compare_arrays(start, end)

    def test_thinning_offcenter(self):
        rowA = [0, 0, 1, 1, 1, 0]
        rowB = [0, 0, 0, 1, 0, 0]
        rowC = [0, 0, 0, 0, 0, 0]
        start, sol = [], []
        for j in range(10):
            start.append(rowA)
            sol.append(rowB)
        start = np.array(start)
        sol = np.array(sol)
        sol[0] = rowC
        sol[-1] = rowC
        sol[-2] = rowC
        self.compare_arrays(sol, skeletonize(start))

if __name__ == '__main__':
    unittest.main()

