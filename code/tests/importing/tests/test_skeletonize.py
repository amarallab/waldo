#!/usr/bin/env python

'''
Filename: 
Discription: 
'''
from SpineProcessing.Code import skeletonize_outline as sk

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

import unittest
import numpy as np

class TestBreakManagement(unittest.TestCase):
    """
    
    compute_skeleton_from_outline
    """
    
    def setUp(self):


        #                               0  1  2  3  4  5  6  7  8
        self.test_pattern1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
                                       [0, 1, 0, 0, 0, 0, 1, 0, 0], # 1
                                       [0, 1, 1, 1, 1, 0, 1, 0, 0], # 2
                                       [0, 0, 0, 0, 0, 1, 1, 0, 0], # 3
                                       [0, 0, 0, 0, 0, 1, 1, 1, 0], # 4
                                       [0, 0, 0, 0, 1, 0, 0, 1, 0], # 5
                                       [0, 0, 0, 1, 0, 0, 0, 1, 0], # 6
                                       [0, 0, 1, 1, 0, 0, 0, 0, 0], # 7
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]) # 8


        # has increased difficulty because some points are on edge
        #                               0  1  2  3  4  5  6  7  8
        self.test_pattern2 = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 0], # 0
                                       [0, 0, 0, 0, 0, 0, 1, 0, 1], # 1
                                       [0, 0, 1, 1, 1, 0, 1, 0, 0], # 2
                                       [0, 0, 0, 0, 0, 1, 1, 0, 0], # 3
                                       [0, 0, 0, 0, 0, 1, 1, 1, 1], # 4
                                       [0, 0, 0, 0, 1, 0, 0, 0, 0], # 5
                                       [0, 0, 0, 1, 0, 0, 0, 0, 0], # 6
                                       [0, 0, 1, 1, 0, 0, 0, 0, 0], # 7
                                       [1, 1, 0, 0, 0, 0, 0, 0, 0]]) # 8


    def test_break_combine_and_seperate(self):
        spine_matrix = self.test_pattern1
        spine_matrix, endpoints = sk.cut_branchpoints_from_spine_matrix(spine_matrix)
        print 'result:'
        print spine_matrix
        
        # copy of code from skeletonize_outline.compute_skeleton_from_outline(outline)
        # TODO rearange functions in skeletonize_outline so that I can test them seperatly.
        #print spine_matrix
        
        #self.assertTrue((len(breakdict1)+len(breakdict2))==len(combined_breaks))
        #self.assertEqual(breakdict1[i], combined_breaks[i])

if __name__ == '__main__':
    unittest.main()









