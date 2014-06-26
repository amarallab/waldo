#!/usr/bin/env python
'''
Filename: distance.py
Description: contains several alternate functions
'''
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import zip

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import math

def compare_functions(input1, input2, labels, function_list):
    import time
    result_list = []
    time_list = []
    for (name, func) in zip(labels, function_list):
        t = time.time()
        result = func(input1, input2)
        t = time.time() - t
        result_list.append(result)
        time_list.append(t)
    print(result_list)
    print(time_list)
    print(max(time_list) / min(time_list))

def euclidean_scipy(u, v):
    from scipy.spatial.distance import euclidean as e
    return e(u, v)

def euclidean_loop(list1, list2):
    dist = 0
    for x, y in zip(list1, list2):
        dist += (x - y) ** 2
    return math.sqrt(dist)

def euclidean(list1, list2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(list1, list2)))

def euclidean_points(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

if __name__ == '__main__':
    input1 = (1, 2)
    input2 = (4, 800)

    #func_list = [euclidean_scipy, euclidean_loop, euclidean, euclidean_points]
    #labels = ['scipy', 'loop', 'final', 'points']
    func_list = [euclidean_scipy, euclidean]
    labels = ['scipy', 'final']
    compare_functions(input1, input2, labels=labels, function_list=func_list)


    # print(euclidean(input1, input2))
    # print(list_euclid(input1, input2))
    # print(list_euclid2(input1, input2))
    # print(static_euclid(input1, input2))
