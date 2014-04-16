#!/usr/bin/env python
'''
Filename: 
Description:


'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import glob
import pandas as pd

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

from annotation.experiment_index import Experiment_Attribute_Index2
from settings.local import LOGISTICS

def count_file_types(basedir):
    counts = {}
    for f in glob.iglob(basedir + '*'):
        bID, filetype = os.path.basename(f).split('-')
        if filetype not in counts:
            counts[filetype] = 0
        counts[filetype] += 1
    return counts


if __name__ == '__main__':
    # toggles
    dataset = 'disease_models'
    dataset = 'N2_aging'
    dataset = 'thermo_recovery'
    data_dir = LOGISTICS['data']

