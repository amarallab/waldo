p#!/usr/bin/env python

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
from importing.datasets import write_combined_worm_percentiles, write_dset_summaries, \
    preprocess_distribution_set
from dsets.check_dset import show_dset, show_dset_completeness
from metrics.measurement_switchboard import FULL_SET, STANDARD_MEASUREMENTS

def main(args):
    for dset in args.dataset:

        data_types = FULL_SET[:]
        if args.run:
            # TODO: write a check to make dset is really a dataset.
            args.dist = args.perc = args.sum = True
        if args.dist:
            preprocess_distribution_set(dset, data_types=data_types)
        if args.sum:
            write_dset_summaries(dset, data_types=data_types)
        if args.perc:
            write_combined_worm_percentiles(dset)

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
    parser.add_argument('--dist', action='store_true', help='show')
    parser.add_argument('--perc', action='store_true', help='show')
    parser.add_argument('--sum', action='store_true', help='show')
    parser.add_argument('--show', action='store_true', help='show')
    parser.add_argument('--check', action='store_true', help='show')
    main(args=parser.parse_args())
