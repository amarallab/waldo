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

def create_backround(impaths):
    """
    create a background image for background subtraction.
    The background image is the maximum pixel values from three grayscale images.

    params
    ---------
    impaths: (list)
       this is a sorted list containing paths to all the image files from one recording.
    """
    first = mpimg.imread(impaths[0])
    mid = mpimg.imread(impaths[int(len(impaths)/2)])
    last = mpimg.imread(impaths[-1])
    return np.maximum(np.maximum(first, mid), last)

def create_binary_mask(img, background, threshold, minsize=100):
    """
    creates a binary array the same size as the image with 1s denoting objects
    and 0s denoting background.

    params
    --------
    img: (image ie. numpy array)
        each pixel denotes greyscale pixel intensities.
    background: (image ie. numpy array)
        the background image with maximum pixel intensities (made with create_background)
    threshold: (float)
        the threshold value used to create the binary mask after pixel intensities for (background - image) have been calculated.
    minsize: (int)
        the fewest allowable pixels for an object. objects with an area containing fewer pixels are removed.       
    """
    mask = (background - img) > threshold
    return morphology.remove_small_objects(mask, minsize)

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

def check_entropy(img):
    """
    for fun, show the entropy for a given image 
    using three different radius sizes.
    
    params
    ------
    img: (image or array)
        the image.
    """
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    ax[0, 0].imshow(img, cmap=plt.cm.Greys_r)
    ax[0, 1].imshow(entropy(img, morphology.disk(3)))
    ax[1, 0].imshow(entropy(img, morphology.disk(5)))
    ax[1, 1].imshow(entropy(img, morphology.disk(20)))


def flip_bbox_xy(bbox):
    """ flips x and y positions for all parts of a bounding box """
    return (bbox[1], bbox[0], bbox[3], bbox[2])

def do_boxes_overlap(box1, box2, show=False):
    """
    returns true if two boxes overlap and false if they do not.
    If the boxes share an edge, then overlap is still true.

    params
    --------
    box1: (tuple of four numbers)
       box corners in form (xmin, ymin, xmax, ymax)
    box2: (tuple of four numbers)
       see box1
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # calculate centroids
    c1 = [(xmax1 + xmin1) / 2.0 , (ymax1 + ymin1) / 2.0] 
    c2 = [(xmax2 + xmin2) / 2.0 , (ymax2 + ymin2) / 2.0] 
    # calculate radius
    r1 = [(xmax1 - xmin1) / 2.0 , (ymax1 - ymin1) / 2.0] 
    r2 = [(xmax2 - xmin2) / 2.0 , (ymax2 - ymin2) / 2.0] 
    # calculate dists between centroids and check against radii
    r12 = np.abs(np.array(c1) - np.array(c2))
    check = (r12 <= np.array(r1) + np.array(r2)).all()
    # show to make sure
    if show:
        fig, ax = plt.subplots()
        rect1 = mpatches.Rectangle((xmin1, ymin1), xmax1 - xmin1, ymax1 - ymin1,
                                   fill=True, alpha=0.3, color='red', linewidth=0.5)
        rect2 = mpatches.Rectangle((xmin2, ymin2), xmax2 - xmin2, ymax2 - ymin2,
                                   fill=True, alpha=0.3, color='blue', linewidth=0.5)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 20])
        plt.show()
    return check

def coordiate_match_offset_arrays(bbox1, array1, bbox2, array2):
    """
    given two image regions with slightly different bounding boxes.
    this function transorms each image so that it is within a common bounding box and
    returns a tuple containing both origional images and the new bounding box coords.

    params
    ------
    bbox1: (tuple of four ints)
       the bounding box for the first image in the form of (xmin, ymin, xmax, ymax)
    array1: (np.array)
       the first image. usually a numpy array containing greyscale pixel intensities.
    bbox2: (tuple of four ints)
       the bounding box for the second image
    array2: (np.array)
       the second image.

    returns
    ------
    new1: (np.array)
       the new version of the first image
    new2: (np.array)
       the new version of the second image
    bbox: (tuple containing four ints)
       the new common bounding box for both images.
    """

    # initialize everything
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    # calculate new bounding box
    mins = np.minimum(np.array(bbox1), np.array(bbox2))[:2]
    maxs = np.maximum(np.array(bbox1), np.array(bbox2))[2:]
    bbox = (mins[0], mins[1], maxs[0], maxs[1])
    xmin, ymin, xmax, ymax = bbox

    # initialize new array shapes
    box_shape = (xmax - xmin + 1, ymax - ymin + 1)
    def fit_old_array_into_new_shape(a, off, shape):
        new = np.zeros(shape, dtype=int)
        xoff, yoff = np.array(off)
        off2 = off + np.array(a.shape)
        xoff2, yoff2 = off2
        #print('new', shape)
        #print(off, list(off2), list(off2 - off))
        #print('x', xoff, ':', xoff2, len(new[xoff:xoff2, 0]))
        #print('broadcast', new[xoff:xoff2, yoff:yoff2].shape, a.shape)
        new[xoff:xoff2, yoff:yoff2] = a
        return new

    # calculate first offsets.    
    #print('array 1')
    offsets = ((xmin1 - xmin), (ymin1 - ymin))
    new1 = fit_old_array_into_new_shape(a=array1, off=offsets, shape=box_shape)
    #print('array 2')
    offsets = ((xmin2 - xmin), (ymin2 - ymin))
    new2 = fit_old_array_into_new_shape(a=array2, off=offsets, shape=box_shape)
    return new1, new2, bbox

# TODO: make this transformation happen.
def filled_image_to_outline_points(bbox, img):
    return 'x', 'y', 'len', 'code'
