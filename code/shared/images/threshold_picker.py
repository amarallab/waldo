# This notebook is for finding the segmentation threshold that most clearly finds worms in a recording.
# It is intended as an alternative method of validating the MultiWorm Tracker's results.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage import morphology
from skimage.measure import regionprops

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(os.path.join(HERE, '..'))
PROJECT_DIR = os.path.abspath(os.path.join(SHARED_DIR, '..'))

sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR)

from code.heltena import profiling

# nonstandard imports
from grab_images import grab_images_in_time_range
from manipulations import create_backround, create_binary_mask
from settings.local import LOGISTICS

MWT_DIR = LOGISTICS['filesystem_data']
print(MWT_DIR)

def show_threshold_properties(img, background, thresholds):
    """
    plots the number, mean area, and std area of objects found for each threshold value specified.

    params
    --------
    img: (image ie. numpy array)
        each pixel denotes greyscale pixel intensities.
    background: (image ie. numpy array)
        the background image with maximum pixel intensities (made with create_background)
    thresholds: (list of floats)
        a list of threshold values to calculate. should be sorted from least to greatest.
    """

    ns, means, stds = [], [], []
    for i, t in enumerate(thresholds):
        # create the masks and calculate the size of all objects found
        mask = create_binary_mask(img, background, threshold=t)
        labels, N = ndimage.label(mask)
        sizes = [r.area for r in regionprops(labels)]
        m, s =  np.mean(sizes), np.std(sizes)
        # if we don't find any objects at this threshold, dont bother calculating for remaining thresholds.
        if len(sizes) == 0:
            thresholds = thresholds[:i]
            break
        ns.append(N)        
        means.append(m)
        stds.append(s)
    final_t = t

    # make the plot
    x = thresholds    
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x, ns, '.--')

    ax[0].set_ylabel('N objects')
    ax[0].set_ylim([0, 150])


    top = np.array(means) + np.array(stds)
    bottom = np.array(means) - np.array(stds)

    ax[1].plot(x, means, '.--', color='blue')
    ax[1].plot(x, top, '--', color='green')
    ax[1].plot(x, bottom, '--', color='green')
        
    ax[1].set_ylabel('mean area')
    ax[0].set_xlim([0, final_t])
    


def show_threshold(img, background, threshold):
    """
    plots an image with the outlines of all objects overlaid on top.

    params
    --------
    img: (image ie. numpy array)
        each pixel denotes greyscale pixel intensities.
    background: (image ie. numpy array)
        the background image with maximum pixel intensities (made with create_background)
    threshold: (float)
        the threshold value used to create the binary mask after pixel intensities for (background - image) have been calculated.
    """ 

    mask = create_binary_mask(img, background, threshold)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.contour(mask, [0.5], linewidths=1.2, colors='b')
    ax.set_title('threshold = {t}'.format(t=threshold))
    ax.axis('off')

def show_threshold_spread(img, background, thresholds=[0.00004, 0.0001, 0.00015, 0.0002]):    
    """
    plots an image four times with the outlines of objects calculated at four different thresholds
    overlaid on them.

    params
    --------
    img: (image ie. numpy array)
        each pixel denotes greyscale pixel intensities.
    background: (image ie. numpy array)
        the background image with maximum pixel intensities (made with create_background)
    thresholds: (list of floats, len=4)
        the threshold values plotted.
    """ 
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, t in enumerate(thresholds):
        row = int(i / 2)
        col = int(i % 2)
        mask = create_binary_mask(img, background, threshold=t)
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
    # list of thresholds to try out
    thresholds = np.linspace(start=0.00001, stop=0.001, num=30)

    # grab images and times.
    times, impaths = grab_images_in_time_range(ex_id, start_time=0)
    times = [float(t) for t in times]
    times, impaths = zip(*sorted(zip(times, impaths)))

    background = create_backround(impaths)

    # pick an image to test. the middle one is good.
    mid = mpimg.imread(impaths[int(len(impaths)/2)])

    # run functions.
    show_threshold_properties(img=mid, background=background, thresholds=thresholds)
    show_threshold_spread(mid, background)
    show_threshold(mid, background, threshold)

    profiling.tag()
    plt.show()
    profiling.end()
