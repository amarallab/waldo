#!/usr/bin/env python

'''
Filename: test_equally_space.py

Description: unit tests to test equally_space.py
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import unittest
from equally_space import *
import matplotlib.pyplot as plt
import numpy as np

class TestEquallySpace(unittest.TestCase):
    def setUp(self):
        pass

    def _make_test_array(self, N, tot_dist):
        # generate N test points randomly spaced between 0 to tot_dist        
        test_array = sorted(np.random.random(N) * tot_dist)
        # set endpoints to be 0 and tot dist
        test_array[0], test_array[-1] = 0, tot_dist
        return test_array

    def assert_arrays_almost_equal(self, a1, a2):
        # works on two numpy arrays of 1 or 2 dimensions
        t1 = a1 - a2
        if isinstance(sum(t1), float):
            self.assertAlmostEqual(sum(t1), 0.0)
            self.assertAlmostEqual(max(t1), 0.0)
            self.assertAlmostEqual(min(t1), 0.0)
        else:
            for i in t1:
                self.assertAlmostEqual(sum(i), 0.0)
                self.assertAlmostEqual(max(i), 0.0)
                self.assertAlmostEqual(min(i), 0.0)
                
    def test_matrix_conversion(self):
        # create numpy arrays
        x = np.arange(12).reshape(3, 4)
        y = np.arange(100, 112).reshape(3, 4)
        # convert back and forth twice
        spines = spine_matricies_to_points(x, y)
        x1, y1 = create_spine_matricies(spines)
        spines2 = spine_matricies_to_points(x1, y1)
        self.assert_arrays_almost_equal(x1, x)
        self.assert_arrays_almost_equal(y1, y)
        for spine1, spine2 in izip(spines, spines2):
            for pt1, pt2 in izip(spine1, spine2):
                self.assertAlmostEqual(pt1, pt2)
        
    def test_equally_space_N(self, show_plot=False):
        N = 10
        tot_dist = 10
        test_points = self._make_test_array(N, tot_dist)
        sp1, sp2 = equally_space_N_xy_points(test_points, test_points, N=N)
        solution_points = np.linspace(0, tot_dist, N)
        if show_plot:
            plt.plot(test_points, lw=0.5, marker='o', label='random')
            plt.plot(sp1, lw=0.5, marker='o', label='spaced1')
            plt.plot(sp2, lw=0.5, marker='o', label='spaced2')
            plt.plot(solution_points, lw=0.5, label='solution')
            plt.legend()
            plt.show()

        self.assert_arrays_almost_equal(sp1, solution_points)
        self.assert_arrays_almost_equal(sp2, solution_points)

# TODO:
# 1. pair this with test_equally_space_N.        
# equally_space_xy_for_stepsize(x, y, step=0.5, kind='linear', n_interp_pts=50)

# 2.
# def set_matrix_orientation(x_mat, y_mat, verbose=True)

# 3.
# smooth_matricies_cols(x_mat, y_mat, window, order)
# smooth_matricies_rows(x_mat, y_mat, window, order)               

# 4.
# equally_space_matrix_distances(x_mat, y_mat)
# equally_space_matricies_times(eq_times, orig_times, x_mat, y_mat)

       
if __name__ == '__main__':
    unittest.main()
