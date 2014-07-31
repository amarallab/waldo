#!/usr/bin/env python
"""prep.

Usage:
  prep.py <command> [options] <id>...
  prep.py (-h | --help)
  prep.py --version

Commands:
  cache            precalculate image threshold values to speed up marking
  mark             manually annotate threshold and ROI images using GUI
  prepare          convert text data into CSVs
  images           use annotated images to calculate
  finish           runs both 'prepare' and 'images'

Arguments:
   id              can be either dataset names or experiment ids

Options:
  -o --overwrite   If data already exists, write over it.
  -h --help        Show this screen.
  --version        Show version.

"""

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
from docopt import docopt
import setpath
os.environ.setdefault('WALDO_SETTINGS', 'default_settings')

from conf import settings
import images
import json
import glob
import prepare
import wio.file_manager as fm
from wio.id_handeling import parse_ids

PREP_DIR = os.path.abspath(settings.LOGISTICS['prep'])
IMAGE_MARK_DIR = os.path.join(PREP_DIR, 'image_markings')
CACHE_DIR = os.path.join(PREP_DIR, 'cache')

def cache_image_data(eids, threshold_file, overwrite=True):
    fm.ensure_dir_exists(CACHE_DIR)
    fm.ensure_dir_exists(IMAGE_MARK_DIR)
    if not eids:
        return
    if not overwrite:
        cached_files = glob.glob(os.path.join(CACHE_DIR, '*'))
        cached_eids = [n.split('cache-')[-1] for n in cached_files]
        cached_eids = [n.split('.json')[0] for n in cached_eids]
        eids_todo = set(eids) - set(cached_eids)
        already_done = set(eids) & set(cached_eids)
        print '{ad} already done'.format(ad=len(already_done))
        eids = list(eids_todo)
    ip = images.InteractivePlot(eids, threshold_file, CACHE_DIR)
    ip.precalculate_threshold_data()

def mark_images_interactivly(eids, threshold_file, overwrite=True):
    fm.ensure_dir_exists(IMAGE_MARK_DIR)
    if not eids:
        return
    if not overwrite:
        thresholds = json.load(open(threshold_file))
        eids_todo = set(eids) - set(thresholds.keys())
        already_done = set(eids) & set(thresholds.keys())
        print '{ad} already done'.format(ad=len(already_done))
        eids = list(eids_todo)
    ip = images.InteractivePlot(eids, threshold_file, CACHE_DIR)
    ip.run_plot()


def reduce_unecessary_work(eids, overwrite, job='prepare'):

    if overwrite:
        return eids

    #TODO a actual check to see if the csvs are already there.
    return eids


if __name__ == '__main__':
    arguments = docopt(__doc__, version='prep 0.1')
    print(arguments)
    eids_by_dset = parse_ids(arguments['<id>'])
    cmd = arguments['<command>']
    overwrite = arguments['--overwrite']
    for dset, eids in eids_by_dset.iteritems():
        print '{cmd}ing {n} recordings for {ds}'.format(ds=dset,
                                                        cmd=cmd,
                                                        n=len(eids))
        filename = 'threshold-{ds}.json'.format(ds=dset)
        threshold_file = os.path.join(IMAGE_MARK_DIR, filename)
        if cmd == 'cache':
            cache_image_data(eids, threshold_file=threshold_file,
                             overwrite=overwrite)
        elif cmd == 'mark':
            mark_images_interactivly(eids, threshold_file=threshold_file,
                             overwrite=overwrite)
        elif cmd == 'prepare':
            eids = reduce_unecessary_work(eids, overwrite, 'prepare')
            for ex_id in eids:
                prepare.summarize(ex_id, verbose=True) #csvs from blob data

        elif cmd == 'images':
            eids = reduce_unecessary_work(eids, overwrite, 'images')
            for ex_id in eids:
                images.summarize(ex_id) #csvs from images

        elif cmd  == 'finish':
            i_eids = reduce_unecessary_work(eids, overwrite, 'images')
            for ex_id in i_eids:
                images.summarize(ex_id) #csvs from images

            p_eids = reduce_unecessary_work(eids, overwrite, 'prepare')
            for ex_id in p_eids:
                images.summarize(ex_id) #csvs from blob data
