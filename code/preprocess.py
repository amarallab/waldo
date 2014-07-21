#!/usr/bin/env python
"""prep.

Usage:
  prep.py <command> <id>...
  prep.py (-h | --help)
  prep.py --version

Commands:
  cache            precalculate image threshold values to speed up marking
  mark             manually annotate threshold and ROI images using GUI
  finish            use annotated images to calculate

Arguments:
   id              can be either dataset names or experiment ids

Options:
  -h --help     Show this screen.
  --version     Show version.

"""

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
#import argparse
from docopt import docopt
import json

import setpath
from images.threshold_picker import InteractivePlot
import images.worm_finder as wf
from annotation.experiment_index import Experiment_Attribute_Index2
import wio.file_manager as fm
from settings.local import LOGISTICS

PREP_DIR = os.path.abspath(LOGISTICS['prep'])
ANNOTATION_DIR = os.path.join(PREP_DIR, 'annotation')
CACHE_DIR = os.path.join(PREP_DIR, 'cache')
fm.ensure_dir_exists(ANNOTATION_DIR)
fm.ensure_dir_exists(CACHE_DIR)

def annotate_images(to_do, threshold_file, precalculate=False):
    """ all arguments are parsed here and the appropriate functions are called.
    :param args: arguments from argparse (namespace object)
    """

    print( 'Error with -c argument')
    print( len(to_do), 'still to go')
    ip = InteractivePlot(to_do, threshold_file, CACHE_DIR)
    if precalculate:
        ip.precalculate_threshold_data()
    else:
        ip.run_plot()

def choose_ex_ids(dset, threshold_file):

    ei = Experiment_Attribute_Index2(dset)
    ex_ids = list(ei.index)
    print len(ex_ids), 'ex_ids in dataset'
    filename = 'threshold-{ds}.json'.format(ds=dset)
    threshold_file = os.path.join(ANNOTATION_DIR, filename)

    #for dset in args.dataset:
    thresholds = {}
    finished_thresholds = set()
    if os.path.isfile(threshold_file):
        thresholds = json.load(open(threshold_file, 'r'))
        finished_thresholds = thresholds.keys()
        print len(finished_thresholds), 'already finished'

    to_do = list(set(ex_ids) - set(finished_thresholds))
    return to_do, list(finished_thresholds)

def short_circuit_preproccessing(ex_ids):
    for ex_id in ex_ids:
        dset = fm.get_dset(ex_id)
        filename = 'threshold-{ds}.json'.format(ds=dset)
        threshold_file = os.path.join(ANNOTATION_DIR, filename)
        print ex_id

        #print json.load(open(threshold_file))[ex_id]
        ip = InteractivePlot([ex_id], threshold_file, CACHE_DIR)
        ip.run_plot()

def finish_preprocessing(annotated_ex_ids):
    """ makes
    - accuracy file
    - matches file
    Main goal:
    - nodenotes file
    """

    for eid in annotated_ex_ids:
        print eid

        pfile = fm.Preprocess_File(ex_id=eid)
        threshold = pfile.threshold()
        roi = pfile.roi()
        print(roi)
        matches, _ = wf.analyze_ex_id_images(eid, threshold, roi)
        break

def main(args):
    to_annotate, annotated = choose_ex_ids(args)
    #annotate_images(to_annotate, args)
    finish_preprocessing(annotated)

#short_circuit_preproccessing(ex_ids=['20130614_120518'])
#short_circuit_preproccessing(ex_ids=['20130426_115023'])

if __name__ == '__main__':
    arguments = docopt(__doc__, version='prep 0.1')
    print(arguments)
    """
    parser = argparse.ArgumentParser(prefix_chars='-',
                                     description="by default it does nothing. but you can specify if it should import, "
                                                 "processes, or aggregate your data.")
    parser.add_argument('dataset', metavar='N', type=str, nargs='+', help='dataset name')
    parser.add_argument('-c', help='configuration username')
    parser.add_argument('-p', action='store_true', help='preprocess')
    main(args=parser.parse_args())
    """
