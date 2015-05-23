#!/usr/bin/env python
'''
Filename: draw_outlines_on_image.py
Description:
'''
from __future__ import absolute_import, division, print_function

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import numpy as np
import glob
import math

# # path definitions
# CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
# SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
# sys.path.append(SHARED_DIR)
# sys.path.append(CODE_DIR)

# nonstandard imports
from Code.waldo.conf import settings
from Code.waldo.encoding.decode_outline import decode_outline

def find_outlines_for_timepoint(ex_id, frame, data_dir=None, overwrite_temp=False):
    """
    returns a list of outlines (where each outline is a list of (x,y) tuples).

    Note:
    This is accomplished by using grep to write a temp file with the results of a regex search.

    :param ex_id: experimentID
    :param frame: the frame for which you would like the outlines (int or numerical string)
    :param data_dir: the directory in which all MWT data is stored.
    """
    if data_dir is None:
        data_dir = settings.LOGISTICS['filesystem_data']
    # specify the name of the temporary file, and use grep to write to it.
    temp_filename = '{dr}{ex_id}/frame{frame}_blobs.tmp'.format(frame=frame, dr=data_dir, ex_id=ex_id)
    print(temp_filename)
    if overwrite_temp or not os.path.exists(temp_filename):
        cmd = 'grep -h \'^{frame}\' {dr}{ex_id}/*.blobs > {tmp_file}'.format(frame=frame, dr=data_dir, ex_id=ex_id,
                                                                             tmp_file=temp_filename)
        os.system(cmd)
    return parse_temp_file_into_outlines(temp_filename)


def parse_temp_file_into_outlines(temp_filename):
    # parse the temp file and only extract encoded outline info.
    with open(temp_filename, 'r') as f:
        blobs = [line.split('%%')[-1].split() for line in f.readlines() if '%%' in line]
    #print(len(blobs), 'blobs found for frame', frame)

    # decode all outlines and convert them into point form (ie. lists of (x,y) tuples)
    outlines = []
    for b in blobs:
        outline_parts = ((b[0], b[1]), b[2], b[3])
        points = decode_outline(outline_parts)
        outlines.append(points)
    return outlines


def create_good_outline_file(ex_id, frame, size_threshold=300, data_dir=None, overwrite_temp=False):

    # test function. keep seperate for easy swapping out.
    def test_sizes(sizes, size_threshold=size_threshold):
        return np.median(sizes) >= size_threshold

    if data_dir is None:
        data_dir = settings.LOGISTICS['filesystem_data']
    frame = str(frame).strip()
    # make sure the specified directory ends in '/'
    if '/' != data_dir[-1]:
        data_dir += '/'
    # if the temp file is already created, dont bother sifting through all the data and creating it again.
    temp_filename = '{dr}{ex_id}/frame{frame}_sizeblobs.tmp'.format(frame=frame, dr=data_dir, ex_id=ex_id)
    #temp_filename = '{dr}frame{frame}_goodblobs.tmp'.format(frame=frame, dr=search_dir)
    if overwrite_temp or not os.path.exists(temp_filename):
        # if no .blobs files are in this directory, the path was probably specified wrong
        files = glob.glob('{path}/*.blobs'.format(path=data_dir + ex_id))
        if len(files) <1:
            print('Warning: {path}\n may not be the correct directory. no blobs files found'.format(path=data_dir+ex_id))

        # sift through all data and grab 'good' outlines
        good_lines = []
        for filename in files:
            with open(filename, 'r') as f:
                local_id, if_good, line_for_frame = None, None, None
                sizes = []
                for line in f:
                    if line[0] == '%':
                        # checking for local id ensures not first line.
                        if local_id and line_for_frame:
                            isGood = test_sizes(sizes)
                            #print(isGood, line_for_frame.split())
                            if isGood:
                                good_lines.append(line_for_frame)
                        local_id = line[1:].strip()
                        sizes = []
                        line_for_frame = None
                    else:
                        sizes.append(int(line.split()[4]))
                        if frame == line[:len(frame)]:
                            line_for_frame = line
        if test_sizes(sizes) and line_for_frame:
            good_lines.append(line)

        print(temp_filename)
        # write the temp file so that we can skip the 'data sifting' step next time.
        with open(temp_filename, 'w') as f:
            for line in good_lines:
                f.write(line)
    # read the temp file and return the contents
    return parse_temp_file_into_outlines(temp_filename)


def find_frame_for_time(ex_id, time, data_dir=None):
    """
    returns the frame and time that is closest to the time specified.

    :param ex_id: experiment ID
    :param time: time in seconds
    :param data_dir: directory in
    """
    if data_dir is None:
        data_dir = settings.LOGISTICS['filesystem_data']
    summary_file = glob.glob('{dr}{ex_id}/*.summary'.format(dr=data_dir, ex_id=ex_id))[0]
    closest_frame, closest_time = 0, 0
    with open(summary_file, 'r') as f:
        for line in f.readlines():
            frame, tp = line.split()[:2]

            if float(tp) > time + 20:
                break
            if math.fabs(float(tp) - time) < math.fabs(float(closest_time) - time):
                closest_frame, closest_time = frame, tp
    return closest_frame, closest_time
