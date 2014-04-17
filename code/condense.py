#!/usr/bin/env python

#TEST

'''
Filename: consolodate_datasets.py
Description: provides a command line user interface with which to consolidate all data
for a particular dataset.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import argparse
import cProfile as profile

# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared/'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from importing.consolidate_dataset import write_combined_worm_percentiles, write_dset_summaries, \
    preprocess_distribution_set
from dsets.check_dset import show_dset, show_dset_completeness

def main(args):    
    for dset in args.dataset:
        if args.run:
            # TODO: write a check to make dset is really a dataset.
            write_dset_summaries(dset)
            write_combined_worm_percentiles(dset)
            preprocess_distribution_set(dset)
        if args.show:
            # show how many recordings/worms we have for each condition
            show_dset(dset)

        if args.check:            
            # show how complete the processing is for dataset
            show_dset_completeness(dset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prefix_chars='-',
                                     description="by default it does nothing. but you can specify if it should import, "
                                                 "processes, or aggregate your data.")
    parser.add_argument('dataset', metavar='N', type=str, nargs='+', help='dataset name')
    parser.add_argument('-c', help='configuration username')
    parser.add_argument('--run', action='store_true', help='show')    
    parser.add_argument('--show', action='store_true', help='show')
    parser.add_argument('--check', action='store_true', help='show')
    main(args=parser.parse_args())    
