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
#import json

import setpath
from conf import settings
import images
#from images.threshold_picker import InteractivePlot
#import images.worm_finder as wf
import collider.prep.prepare as prep

#from annotation.experiment_index import Experiment_Attribute_Index2
import wio.file_manager as fm

PREP_DIR = os.path.abspath(settings.LOGISTICS['prep'])
IMAGE_MARK_DIR = os.path.join(PREP_DIR, 'image_markings')
CACHE_DIR = os.path.join(PREP_DIR, 'cache')


def cache_image_data(eids, threshold_file):
    fm.ensure_dir_exists(CACHE_DIR)
    fm.ensure_dir_exists(IMAGE_MARK_DIR)
    ip = images.InteractivePlot(eids, threshold_file, CACHE_DIR)
    ip.precalculate_threshold_data()

def mark_images_interactivly(eids, threshold_file):
    fm.ensure_dir_exists(IMAGE_MARK_DIR)
    ip = images.InteractivePlot(eids, threshold_file, CACHE_DIR)
    ip.run_plot()

def finish_preprocessing(eids):
    for ex_id in eids:
        prep.summarize(ex_id) #csvs from blob data
        images.summarize(ex_id) #csvs from image data

#TODO make this be able to accept both
def parse_ids(ids):
    dset_eids = {}
    for eid in ids:
        dset = fm.get_dset(eid)
        if dset not in dset_eids:
            dset_eids[dset] = []
        dset_eids[dset].append(eid)
    #print(dset_eids)
    return dset_eids


if __name__ == '__main__':
    arguments = docopt(__doc__, version='prep 0.1')
    #print(arguments)
    dset_eids = parse_ids(arguments['<id>'])
    cmd = arguments['<command>']
    for dset, eids in dset_eids.iteritems():
        print '{cmd}ing {n} recordings for {ds}'.format(ds=dset,
                                                        cmd=cmd,
                                                        n=len(eids))
        filename = 'threshold-{ds}.json'.format(ds=dset)
        threshold_file = os.path.join(IMAGE_MARK_DIR, filename)
        print threshold_file
    if cmd == 'cache':
          cache_image_data(eids, threshold_file=threshold_file)
    if cmd == 'mark':
          mark_images_interactivly(eids, threshold_file=threshold_file)
    if cmd  == 'finish':
          finish_preprocessing(eids)
