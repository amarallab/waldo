#!/usr/bin/env python

'''
Filename: import_test_worms.py
Discription: This script contains all necessary functions to import jsons containing test data into the basic
input types used for all data processing code.

The jsons are not generated using this code, but they contain dictionaries using times as keys
(bson friendly formatting) and lists of x,y tuples that specify points along a worm's outline as values)
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import json
import os
import sys
from glob import glob
import numpy as np

# path specifications
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
shared_code_directory = project_directory + 'Shared/Code'
test_directory = project_directory + 'SideProjects/TestWorms/'
test_plot_dir = test_directory + 'Results/'
assert os.path.exists(project_directory), 'project directory not found'
assert os.path.exists(test_plot_dir)
assert os.path.exists(shared_code_directory), 'Shared/Code directory not found'
sys.path.append(shared_code_directory)

# nonstandard imports
from Shared.Code.Plotting.SingleWorms.single_worm_suite import single_worm_suite
from import_test_worms import import_tests_of_type

# globals
key_plots_for_test_type = {'SquareTests': ['centroid']}

if __name__ == '__main__':
    test_type = 'SquareTests'
    save_dir = test_plot_dir + test_type + '/'
    print save_dir
    test_blobs = import_tests_of_type(test_type=test_type, import_data=False)
    for blob_id in test_blobs:
        single_worm_suite(blob_id=blob_id, save_dir=save_dir)