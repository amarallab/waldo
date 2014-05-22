# This notebook is for finding the segmentation threshold that most clearly finds worms in a recording.
# It is intended as an alternative method of validating the MultiWorm Tracker's results.

# standard imports
from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

import os
import sys
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import prettyplotlib as ppl
import random
import functools

#import Image
import numpy as np
import scipy
from scipy import ndimage
from skimage import morphology
from skimage.measure import regionprops

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../shared/')
PROJECT_DIR = os.path.abspath(HERE + '/../../')
sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR)

from code.heltena import profiling

# nonstandard imports
from images.grab_images import grab_images_in_time_range
#from wio.file_manager import get_good_blobs, get_timeseries
from settings.local import LOGISTICS

MWT_DIR = LOGISTICS['filesystem_data']
print(MWT_DIR)

def create_backround(impaths):
    first = mpimg.imread(impaths[0])
    mid = mpimg.imread(impaths[int(len(impaths)/2)])
    last = mpimg.imread(impaths[-1])
    return np.maximum(np.maximum(first, mid), last)

def find_objects(img, background, threshold=0.0001, minsize=100):
    mask = (background - img) > threshold
    mask2 = morphology.remove_small_objects(mask, minsize)
    labels, n_features = ndimage.label(mask2)
    return labels, n_features, mask2

def pick_threshold_in_range(img, background, trange=[0.00001, 0.001], num=80, show=True, best_threshold=0.001):
    space = np.linspace(start=0.00001, stop=0.001, num=num)
    ns, means, meds, stds = [], [], [], []
    maxs, mins = [], []
    for i, t in enumerate(space):
        objects, N, mask = find_objects(img, background, threshold=t)
        sizes = [r.area for r in regionprops(objects)]
        m, s, md =  np.mean(sizes), np.std(sizes), np.median(sizes)
        if len(sizes) == 0:
            space = space[:i]
            break
        maxs.append(scipy.stats.scoreatpercentile(sizes, 90))
        mins.append(scipy.stats.scoreatpercentile(sizes, 10))
        print(t, 'N:', N, 'mean size:', m, 'std:', s)
        ns.append(N)        
        means.append(m)
        meds.append(md)
        stds.append(s)

    final_t = t

    # TODO: GET RID OF THIS VALUE
    bt = best_threshold

    if show:
        x = space
        print(len(x), len(ns), len(meds), len(maxs))

        fig, ax = plt.subplots(4, 1, sharex=True)
        ax[0].plot(x, ns, '.--')
        ax[0].plot([bt, bt], [min(ns), max(ns)], color='red')

        ax[0].set_ylabel('N objects')
        ax[0].set_ylim([0, 150])
        ax[1].plot(x, means, '.--', label='mean')
        ax[1].plot(x, meds, '.--', label='median')
        ax[1].plot(x, maxs, '.--', label='90th')
        ax[1].plot(x, mins, '.--', label='10th')
        ax[1].legend(loc='upper right')
        ax[1].plot([bt, bt], [min(means), max(means)], color='red')
        ax[1].set_ylabel('area')

        ran = np.array(maxs) - np.array(mins)
        ax[2].plot(x, stds, '.--', label='std')
        ax[2].plot(x[:len(ran)], ran, '.--', label='10th to 90th range')
        ax[2].plot([bt, bt], [min(stds), max(stds)], color='red')
        ax[2].set_ylabel('std area')
        ax[2].legend(loc='upper right')


        rat = np.array(means) / np.array(stds)
        ax[3].plot(x[:len(rat)], rat, '.--', label= 'mean / std')

        rat2 = np.array(meds) / np.array(ran)
        ax[3].plot(x[:len(rat2)], rat2, '.--', label= 'med / range')

        rat3 = np.array(meds) / np.array(stds)
        ax[3].plot(x[:len(rat3)], rat3, '.--', label= 'med / std')

        ax[3].plot([bt, bt], [min(rat), max(rat)], color='red')
        ax[3].set_ylabel('area mean/std')
        ax[3].set_xlabel('threshold')
        ax[3].legend(loc='upper left')
        ax[0].set_xlim([0, final_t])
    
    return best_threshold


def show_threshold(img, background, threshold):
    objects, N, mask = find_objects(img, background, threshold)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax1.contour(mask, [0.5], linewidths=1.2, colors='b')
    ax2.imshow(objects, cmap=plt.cm.jet, interpolation='nearest')
    ax2.set_title('threshold = {t}'.format(t=threshold))
    ax1.axis('off')
    ax2.axis('off')


def show_threshold_spread(img, background, thresholds=[0.00004, 0.0001, 0.00015, 0.0002]):

    
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, t in enumerate(thresholds):
        row = int(i / 2)
        col = int(i % 2)

        objects, N, mask = find_objects(img, background, threshold=t)
        ax[row, col].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
        ax[row, col].contour(mask, [0.5], linewidths=1.2, colors='b')
        ax[row, col].set_title('threshold = {t}'.format(t=t))
        ax[row, col].axis('off')

if __name__ == '__main__':
    ex_id = '20130610_161943'
    #ex_id = '20130318_131111'
    threshold = 0.0001
    threshold = 0.0003

    profiling.begin()
    # grab images and times.
    times, impaths = grab_images_in_time_range(ex_id, start_time=0)
    times = [float(t) for t in times]
    times, impaths = zip(*sorted(zip(times, impaths)))

    background = create_backround(impaths)

    mid = mpimg.imread(impaths[int(len(impaths)/2)])
    #threshold = pick_threshold_in_range(img=mid, background=background)
    show_threshold_spread(mid, background)
    #show_threshold(mid, background, threshold)

    profiling.tag()
    plt.show()
    profiling.end()
