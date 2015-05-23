from __future__ import absolute_import, division, print_function
# This notebook is for finding the segmentation threshold that most clearly finds worms in a recording.
# It is intended as an alternative method of validating the MultiWorm Tracker's results.

# standard library
# import sys
# import os

# third party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import matplotlib.cm as cm
from scipy import ndimage
from skimage import morphology
# from skimage.measure import regionprops
# from skimage.filter.rank import entropy
import matplotlib.patches as mpatches

# package specific

def outline_to_outline_matrix(outline, bbox=None):
    """
    returns a filled in binary image of a list of outline points

    params
    -----
    outline: (list of tuples)
       the list of points to be turned into a filled in image [(x1, y1), (x2, y2) ... etc.]
    bbox: (tuple of four ints)
       the bounding box for the image to be created in the form of (xmin, ymin, xmax, ymax)
       if not specified, just takes smallest bounding box around outline points.
    returns
    ------
    outline_matrix: (np.array)
        an np array containing boolean values denoting the filled in outline shape.
    """
    if len(outline) == 4:
        print('precaution, a len 4 outline is usually something else by accident')
        print(outline)
    # prepare blob outline and bounding box.
    if isinstance(outline, np.ndarray):
        x = outline[:, 0]
        y = outline[:, 1]
    else:
        x, y = zip(*outline)

    if bbox == None:
        bbox = (min(x), min(y), max(x), max(y))
    minx, miny, maxx, maxy = bbox
    x = [i - minx for i in x]
    y = [i - miny for i in y]
    shape = (maxx - minx + 1, maxy - miny + 1)
    outline_matrix = np.zeros(shape)
    for i, j in zip(x, y):
        # print(i, j)
        outline_matrix[i, j] = 1
    return ndimage.morphology.binary_fill_holes(outline_matrix)


def create_roi_mask(x, y, r, shape):
    nx, ny = shape
    xs = np.arange(0, nx)
    ys = np.arange(0, ny)
    xv, yv = np.meshgrid(xs, ys)
    dy = yv - y
    dx = xv - x
    d = np.sqrt(dy ** 2 + dx ** 2).T
    roi_mask = d <= r
    return roi_mask


def create_backround(impaths):
    """
    create a background image for background subtraction.
    The background image is the maximum pixel values from three grayscale
    images.

    params
    ---------
    impaths: (list)
       this is a sorted list containing paths to all the image files from one
       recording.
    """
    first = mpimg.imread(impaths[0])
    mid = mpimg.imread(impaths[int(len(impaths) / 2)])
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
    c1 = [(xmax1 + xmin1) / 2.0, (ymax1 + ymin1) / 2.0]
    c2 = [(xmax2 + xmin2) / 2.0, (ymax2 + ymin2) / 2.0]
    # calculate radius
    r1 = [(xmax1 - xmin1) / 2.0, (ymax1 - ymin1) / 2.0]
    r2 = [(xmax2 - xmin2) / 2.0, (ymax2 - ymin2) / 2.0]
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


def points_to_aligned_matrix(outline_points):
    """
    this function takes a list of outlines (in point form)
    and makes filled binary matrices for all of them in the same
    coordinate system.

    params
    -----
    outline_points: (list of lists)
    each list in outline_points is a list of xy tuples for the contour
    of a blob.

    returns
    -----
    aligned_matricies: (list of ndarrays)
    bbox: (tuple of four ints)
        bounding box for all arrays in the form (xmin, ymin, xmax, ymax)
    """
    outline_matricies, bboxes = [], []
    for outline in outline_points:
        x, y = zip(*outline)
        bboxes.append((min(x), min(y), max(x), max(y)))
        outline_matricies.append(outline_to_outline_matrix(outline))
    aligned_matricies, bbox = align_outline_matricies(outline_matricies, bboxes)
    return aligned_matricies, bbox


def align_outline_matricies(outline_matricies, bboxes):
    """
    aligns a list of outline matricies so that all of them are on the same coordinate system.

    params
    ------
    outline_matricies: (list)
       each outline matrix is a binary np.ndarray containing the shape of a blob. (1 = blob, 0 = background)
    bboxes: (list)
       this list contains bounding boxes corresponding to each of the outline matricies.
       a bounding box consists of a tuple of four ints (min(x), min(y), max(x), max(y))

    returns
    ------
    aligned_matricies: (list of np.ndarrays)
        the list of outline matricies all aligned onto a common coordinate system
    bbox: (tuple containing four ints)
       the new common bounding box for all images
    """
    aligned_matricies = []
    primary_matrix = outline_matricies[0]
    primary_bbox = bboxes[0]

    # fig, ax = plt.subplots()
    # ax.imshow(primary_matrix)

    # the loop creates a primary with a bbox that encompasses all other boxes.
    for om, bb in zip(outline_matricies[1:], bboxes[1:]):
        primary_matrix, om, primary_bbox = coordiate_match_offset_arrays(primary_bbox, primary_matrix, bb, om)
        # print(primary_bbox)
    aligned_matricies = [primary_matrix]

    # fig, ax = plt.subplots()
    # ax.imshow(primary_matrix)
    # plt.show()

    # the loop ensures all boxes match the primary.
    for om, bb in zip(outline_matricies[1:], bboxes[1:]):
        primary_matrix, om, primary_bbox = coordiate_match_offset_arrays(primary_bbox, primary_matrix, bb, om)
        aligned_matricies.append(om)
        # print(primary_bbox)
    return aligned_matricies, primary_bbox


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
        # print('new', shape)
        # print(off, list(off2), list(off2 - off))
        # print('x', xoff, ':', xoff2, len(new[xoff:xoff2, 0]))
        # print('broadcast', new[xoff:xoff2, yoff:yoff2].shape, a.shape)
        new[xoff:xoff2, yoff:yoff2] = a
        return new

    # calculate first offsets.
    # print('array 1')
    offsets = ((xmin1 - xmin), (ymin1 - ymin))
    new1 = fit_old_array_into_new_shape(a=array1, off=offsets, shape=box_shape)
    # print('array 2')
    offsets = ((xmin2 - xmin), (ymin2 - ymin))
    new2 = fit_old_array_into_new_shape(a=array2, off=offsets, shape=box_shape)
    return new1, new2, bbox


# TODO: make this transformation happen.
def filled_image_to_outline_points(bbox, img):
    return 'x', 'y', 'len', 'code'
