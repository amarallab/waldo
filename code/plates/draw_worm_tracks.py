#!/usr/bin/env python
'''
Filename: draw_plate_worm_tracks.py

Description:
Draws all the tracks of worms for a given plate onto one image from that plate.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.file_manager import format_results_filename, get_good_blobs, get_timeseries, get_dset
from images.grab_images import get_base_image_path
from importing.centroid import fill_gaps

def get_all_worm_tracks(ex_id):
    ''' returns a list of blob_ids, a string denoting the dataset, and a list of lists (xy positions)
    for all blobs recorded for ex_id.
    '''

    blobs = get_good_blobs(ex_id, 'xy_raw')
    dset = get_dset(ex_id)
    tracks = []
    print '{ID}: {N} blobs found'.format(ID=ex_id, N=len(blobs))
    for blob_id in blobs:
        t, xy = get_timeseries(blob_id, data_type='xy_raw')
        xy = fill_gaps(xy)
        tracks.append(xy)
    return blobs, dset, tracks

def draw_plate_tracks(ex_id, blobs=[], save=True):
    ''' draws all blob tracks from ex_id onto one image.
    '''

    blobs2, dset, tracks2 = get_all_worm_tracks(ex_id)

    if len(blobs2) == 0:
        print 'no blobs found. exiting'
        return None

    if len(blobs):
        nblobs = []
        tracks = []
        for b, t in zip(blobs2, tracks2):
            if b in blobs:
                nblobs.append(b)
                tracks.append(t)
        blobs = nblobs

    else:
        blobs = blobs2
        tracks = tracks2

    im_path = get_base_image_path(ex_id)
    if im_path == None:
        print 'no images found for plate: {ID}'.format(ID=ex_id)
        return
    img = Image.open(im_path)
    ymax, xmax = img.size
    background = np.array(img)#.T #.reshape(img.size[0], img.size[1])


    fig, ax = plt.subplots()
    ax.imshow(background.T, cmap=cm.Greys_r)

    colormap = cm.spectral
    ax.set_color_cycle([colormap(i) for i in
                        np.linspace(0, 0.9, 12)])
    for bID, xy in zip(blobs,tracks):
        if xy!=None:
            if len(xy) > 1:
                x, y = zip(*xy)
                ax.plot(x,y)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(0.0)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(0.0)

    plt.xlim([0, xmax])
    plt.ylim([0, ymax])
    plt.show()

    if not save:
        return
    ID = ex_id
    result_type='multi_path'
    tag= None
    dset=dset
    ID_type='p'
    save_name = format_results_filename(ID, result_type, tag,
                                        dset, ID_type, ensure=True)
    print save_name
    plt.savefig(save_name)
    plt.clf()


if __name__ == '__main__':
    ex_id = '20130614_120518'
    #blobs = ['20130614_120518_02263', '20130614_120518_03221', '20130614_120518_00910', '20130614_120518_02765', '20130614_120518_00020']
    blobs = []
    ex_id = '20130318_131111'
    ex_id = '20130614_120518'
    draw_plate_tracks(ex_id, blobs=blobs)
