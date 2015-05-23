#!/usr/bin/env python
from __future__ import absolute_import, print_function
'''
Filename: grab_images.py
Description: functions for finding and returning particular aspects of the MWT raw data such as images or outlines for
a particular time.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard library
import sys
import os
import math
import pathlib

# third party
import numpy as np
#import Image

# package imports
from waldo.conf import settings
from waldo.wio import paths
from .grab_outlines import find_frame_for_time

def _datadir(ex_id, data_dir):
    if data_dir is None:
        data_dir = paths.experiment(ex_id)
    else:
        data_dir = pathlib.Path(data_dir)
    return data_dir

def create_image_directory(experiment, data_dir=None):
    """
    returns a dictionary containing all the times at which images were taken and the paths to those images.

    :param ex_id: experiment ID
    :param data_dir: the directory in which all MWT data is stored.
    :return: a dictionary with times (keys) and the path to the image taken at that time (value)
    """
    #images = get_images(ex_id, data_dir)
    if not images:
        print('something may be wrong with search path. no images found: ',
              str(_datadir(ex_id, data_dir)))
        return None
    basename = ['z' for _ in range(1000)]
    for i in images:
        if len(i) < len(basename):
            basename = i
    basename = basename.split('.png')[0]

    time_to_image = {}
    for i in images:
        time = i.split(basename)[-1].split('.png')[0]
        if time:
            if time + '.png' == i:
                print('i', i)
                print('base', basename)
                print('time', time)
                continue
            #time = '%.3f' % (int(time) / 1000.0)
            time = '{t}'.format(t=round(int(time) / 1000.0), ndigits=3)
        else:
            time = '0.000'
        time_to_image[time] = i
    return time_to_image


# def crop_image_around_worm(image_file, xy_shift, xy_size, margin=0):
#     """
#     create a small image of the region directly surrounding one worm.
#
#     :param image_file: the path to the image which contains the worm
#     :param xy_shift: coordinates for the corner of the box to be cropped around the worm in pixels (tuple of two ints)
#     :param xy_size: the size of the box to be cropped around the worm in pixels (tuple of two ints)
#     :param margin: the number of
#     :return: a numpy array of the cropped worm image.
#     """
#     im = Image.open(image_file)
#     x0, y0 = xy_shift
#     x1, y1 = xy_size
#     x1 += x0
#     y1 += y0
#     #box = (y0 - margin, x0 - margin, y1 + margin, x1 + margin)
#     region = im.crop((y0 - margin, x0 - margin, y1 + margin, x1 + margin))
#     return np.asarray(region)
#
# if __name__ == '__main__':
#     ex_id = '20130323_170511'
#     image_times = create_image_directory(ex_id=ex_id)
#     closest_image_time, closest_image = get_closest_image(target_time=500, image_dict=image_times)
#     frame, timepoint = find_frame_for_time(ex_id=ex_id, time=closest_image_time)
#     data_dir = settings.LOGISTICS['filesystem_data']
#     temp_filename = '{dr}{ex_id}/frame{frame}_blobs.tmp'.format(frame=frame, dr=data_dir, ex_id=ex_id)
#     with open(temp_filename, 'r') as f:
#         for line in f:
#             print(line)
