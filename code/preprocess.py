#!/usr/bin/env python

#TEST

'''
Filename: preprocess.py
Description: provides a command line user interface with which to process data.
'''
__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import argparse
import json

# path definitions
'''
CODE_DIR = os.path.dirname('.')
PROJECT_DIR = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = os.path.join(CODE_DIR, 'shared')
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)
'''

import setpath
from images.threshold_picker import InteractivePlot
from annotation.experiment_index import Experiment_Attribute_Index2
from wio.file_manager import ensure_dir_exists, get_dset
from settings.local import LOGISTICS

PRETREATMENT_DIR = os.path.abspath(LOGISTICS['pretreatment'])
CACHE_DIR = os.path.join(PRETREATMENT_DIR, 'cache')
ensure_dir_exists(PRETREATMENT_DIR)
ensure_dir_exists(CACHE_DIR)

def main(args):
    """ all arguments are parsed here and the appropriate functions are called.
    :param args: arguments from argparse (namespace object)
    """
    if args.c is not None:
        print 'Error with -c argument'



    for dset in args.dataset:
        print dset
        ei = Experiment_Attribute_Index2(dset)
        ex_ids = list(ei.index)
        print len(ex_ids), 'ex_ids in dataset'

        filename = 'threshold-{ds}.json'.format(ds=dset)
        threshold_file = os.path.join(PRETREATMENT_DIR, filename)
        thresholds = {}
        if os.path.isfile(threshold_file):
            thresholds = json.load(open(threshold_file, 'r'))
            finished_thresholds = thresholds.keys()
            print len(finished_thresholds), 'already finished'

        to_do = list(set(ex_ids) - set(finished_thresholds))
        print len(to_do), 'still to go'
        ip = InteractivePlot(to_do, threshold_file, CACHE_DIR)
        if args.p:
            ip.precalculate_threshold_data()
        else:
            ip.run_plot()

def short_circuit_preproccessing(ex_ids):
    for ex_id in ex_ids:
        dset = get_dset(ex_id)
        filename = 'threshold-{ds}.json'.format(ds=dset)
        threshold_file = os.path.join(PRETREATMENT_DIR, filename)
        print ex_id

        #print json.load(open(threshold_file))[ex_id]
        ip = InteractivePlot([ex_id], threshold_file, CACHE_DIR)
        ip.run_plot()

short_circuit_preproccessing(ex_ids=['20130614_120518'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prefix_chars='-',
                                     description="by default it does nothing. but you can specify if it should import, "
                                                 "processes, or aggregate your data.")
    parser.add_argument('dataset', metavar='N', type=str, nargs='+', help='dataset name')
    parser.add_argument('-c', help='configuration username')
    parser.add_argument('-p', action='store_true', help='preprocess')
    main(args=parser.parse_args())
