#!/usr/bin/env python

'''
Filename: create_fake_blobdict_json.py
Discription: creates a json with a dictionary of attributes found in regular worm data.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import json
import os
#import sys
#import numpy as np
#from glob import glob
# nonstandard imports
test_directory = os.path.dirname(os.path.abspath(__file__))
assert os.path.exists(test_directory), 'test directory not found'
#sys.path.append(code_directory)

def create_fake_blobdict_json(save_dir=test_directory + '/Settings'):
    '''
    '''
    print save_dir
    assert os.path.isdir(save_dir)
    fake_dict = {# the commented out datatypes must be specified durint the creation of the fakeworm data
                 #'data'
                 #'description'
                 #'blobid'
                 'part': '1of1',
                 'ex_id': '00000000_000001',
                 'vid_flags': '',
                 'name': 'test worm',
                 'source_camera': 'none',
                 'purpose': 'code_testing',
                 'strain': 'test',
                 'age': 'test',
                 'blob_flag': True,
                 'growth_medium': 'test',
                 'set_temp': 'test',
                 'stimulous': 'test',
                 'food': 'test',
                 'compounds': 'test',
                 'lid?': 'test',
                 'vid_duration': 'test',
                 'num_blobs_files': 0,
                 'num_images': 0,
                 'zoom': 'test',
                 'bl_dist': 5,
                 'pixels_per_mm': 45,}
    json.dump(fake_dict, open(save_dir+'/fake_dict.json', 'w'))

if __name__ == '__main__':
    create_fake_blobdict_json()
