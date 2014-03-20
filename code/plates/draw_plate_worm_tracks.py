#!/usr/bin/env python
'''
Filename: process_plate_timeseries.py
Description:
Pull one type of data out of database, and save it in jsons organized by ex_id.
data pulled is broken into 15 minute segments. within each 15min segment data is pulled either
by subsampling the data or by binning into 10 second bins.
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
#from exponential_fitting import fit_exponential_decay_robustly, rebin_data, exponential_decay, fit_constrained_decay_in_range
#from plate_utilities import get_ex_id_files,  write_dset_summary, parse_plate_timeseries_txt_file
#from plate_utilities import return_flattened_plate_timeseries, organize_plate_metadata
from wio.file_manager import format_results_filename, get_good_blobs, get_timeseries, get_dset
from imagehandeling.grab_images import get_base_image_path

def get_all_worm_tracks(ex_id):
    blobs = get_good_blobs(ex_id)
    dset = get_dset(ex_id)
    tracks = []
    print '{ID}: {N} blobs found'.format(ID=ex_id, N=len(blobs))
    for blob_id in blobs:
        t, xy = get_timeseries(blob_id, data_type='xy_raw', ID_type='w')
        tracks.append(xy)
    return blobs, dset, tracks

def draw_plate_tracks(ex_id,save=True):
    

    blobs, dset, tracks = get_all_worm_tracks(ex_id)
    if len(blobs) == 0:
        print 'no blobs found. exiting'
        return None

    im_path = get_base_image_path(ex_id)
    img = Image.open(im_path)
    ymax, xmax = img.size
    background = np.array(img)#.T #.reshape(img.size[0], img.size[1])


    fig, ax = plt.subplots()
    ax.imshow(background.T, cmap=cm.Greys_r)

    colormap = cm.spectral
    ax.set_color_cycle([colormap(i) for i in 
                        np.linspace(0, 0.9, 12)])
    for bID, xy in zip(blobs,tracks):
        if xy:
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
    draw_plate_tracks(ex_id)
