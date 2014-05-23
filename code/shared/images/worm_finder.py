# This notebook is for exploring methods for finding worms in a recording's images.
# It is intended as an alternative method of validating the MultiWorm Tracker's results.

# standard imports
from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

import os
import sys
import itertools

import random
import functools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import prettyplotlib as ppl

import scipy
from scipy import ndimage
import skimage
from skimage import morphology
from skimage.measure import regionprops
from skimage.filter.rank import entropy

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../shared/')
PROJECT_DIR = os.path.abspath(HERE + '/../../')
sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR)

# nonstandard imports
from manipulations import create_backround, create_binary_mask, show_threshold
from manipulations import coordiate_match_offset_arrays, do_boxes_overlap, filled_image_to_outline_points
from grab_images import grab_images_in_time_range
from wio.file_manager import get_good_blobs, get_timeseries, ensure_dir_exists #write_table , get_dset, read_table
from joining import multiworm
from joining.multiworm.readers import blob as blob_reader
from settings.local import LOGISTICS

MWT_DIR = os.path.abspath(LOGISTICS['filesystem_data'])
VALIDATION_DIR = os.path.abspath(LOGISTICS['validation'])


# Derived from http://stackoverflow.com/a/2566508/194586
# However, I claim these as below the threshold of originality
def find_nearest_index(seq, value):
    return (np.abs(np.array(seq)-value)).argmin()

def frame_parser(blob_lines, frame):
    """
    A lighter, probably quicker, parser to just get a single frame of 
    data out of a blob.
    """
    first_line = six.next(blob_lines)
    frame_offset = frame - int(first_line.split(' ', 1)[0])
    line = first_line

    # blindly consume as many lines as needed
    try:
        for dummy in range(frame_offset):
            line = six.next(blob_lines)
    except MWTDataError:
        pass

    # parse the line and return
    blob = blob_reader.parse([line])
    if blob['frame'][0] != frame:
        raise multiworm.core.MWTDataError("Blob line offset failure")
    return blob

def frame_parser_spec(frame):
    return functools.partial(frame_parser, frame=frame)
 
def outline_to_oubline_matrix(bbox, outline):
    """
    returns a filled in binary image of a list of outline points

    params
    -----
    bbox: (tuple of four ints)
       the bounding box for the image to be created in the form of (xmin, ymin, xmax, ymax)
    outline: (list of tuples)
       the list of points to be turned into a filled in image [(x1, y1), (x2, y2) ... etc.]

    returns
    ------
    outline_matrix: (np.array)
        an np array containing boolean values denoting the filled in outline shape.
    """
    
    minx, miny, maxx, maxy = bbox
    x, y = zip(*outline)
    x = [i - minx for i in x]
    y = [i - miny for i in y]
    shape = (maxx - minx + 1, maxy - miny + 1)
    outline_matrix = np.zeros(shape)
    for i, j in zip(x, y):
        #print(i, j)
        outline_matrix[i, j] = 1
    return ndimage.morphology.binary_fill_holes(outline_matrix)

def match_objects(bids, blob_centroids, blob_outlines, image_objects, maxdist=20, verbose=False):
    """
    workhorse function, working hard so that you don't have to.
    """

    # initialize everything.
    img_centroids = np.array([r.centroid for r in image_objects])
    #blob_centroids = np.array(blob_centroids)
    matches, failures, lines = [], [], []
    blobs_by_object = {}
    key_outlines = {}
    for bid, c1, outline in zip(bids, blob_centroids, blob_outlines):
        
        # prepare blob outline and bounding box.
        x, y = zip(*outline)
        blob_bbox = (min(x), min(y), max(x), max(y))
        # calculate distances to all image object centroids.
        dx = img_centroids[:, 0] - c1[0]
        dy = img_centroids[:, 1] - c1[1]
        dists = np.sqrt(dx** 2 + dy** 2)

        # initialize dummy variables and loop over image objects.
        closest_dist = 10 *  maxdist
        closest_obj = -1
        overlap_exists = False
        #for i, d in enumerate(dists):
        for im_obj, d in zip(image_objects, dists):
            if d < maxdist and d < closest_dist:
                # ok blob is sufficiently close, test if bounding boxes overlap.
                #img_bbox = img_bboxes[i]
                img_bbox = im_obj.bbox
                # if boxes overlap, store match.
                if do_boxes_overlap(img_bbox, blob_bbox):
                    closest_obj = im_obj
                    closest_cent = im_obj.centroid
                    closest_dist = d                

        if closest_obj != -1:
            # we have a single closest match for this blob. lets do the calculations.
            # the final validation step: bid outline must have more overlapping pixels than overreaching.
            # ie. object must be mostly on top of the image_object
            outline_mat = outline_to_oubline_matrix(blob_bbox, outline)
            obj_bbox, obj_img = closest_obj.bbox, closest_obj.image
            outline_arr, img_arr, bbox = coordiate_match_offset_arrays(blob_bbox, outline_mat,
                                                                       obj_bbox, obj_img)
            img_arr = img_arr * 2
            overlay = img_arr + outline_arr
            # keep just to look every once in a while.
            if False:
                fig, ax = plt.subplots(1,3)
                ax[0].imshow(outline_arr)
                ax[1].imshow(img_arr)
                ax[2].imshow(overlay)
                plt.show()

            # calculate pixel matches.
            overlaps = (overlay == 3).sum()
            underlaps = (overlay == 2).sum()
            overreaches = (overlay == 1).sum()

            # if the objects are mostly on top of one another: count as validated match.
            if overlaps > overreaches:
                # this blob is officially validated.
                matches.append(bid)                
                
                # save for potential joins.
                l = closest_obj.label
                if l not in blobs_by_object:                
                    blobs_by_object[l] = []
                blobs_by_object[l].append(bid)

                # save a connecting line for visual validation.
                xs = [c1[0], closest_cent[0]]
                ys = [c1[1], closest_cent[1]]
                lines.append((xs, ys))

                # Todo: make this real data.
                key_outlines[bid] = filled_image_to_outline_points(obj_bbox, obj_img)

        else:
            # this is officially a bad blob.
            failures.append(bid)

    # reformat joins
    blobs_to_join = []
    for l in blobs_by_object.values():
        if len(l) > 1:
            blobs_to_join.append(l)

    if verbose:
        print(len(blob_centroids), 'blobs tracked by MWT')
        print(len(failures), 'blobs without matches')
        print(len(matches), 'blobs matched to image objects')

    return matches, failures, blobs_to_join, key_outlines, lines



def grab_blob_data(experiment, time):
    """
    pull the frame number and a list of tuples (bid, centroid, outline)
    for a given time and experiment.

    params
    -------
    experiment: (experiment object from multiworm)
        cooresponds to a specific ex_id
    time: (float)
        the closest time in seconds for which we would like to retrieve data

    returns
    -------
    frame: (int)
        the frame number that most closely matches the given time.
    blob_data: (list of tuples)
        the list contains the (blob_id [str], centroid [xy tuple], outlines [list of points]) 
        for all blobs tracked during that particular frame.
    """

    # get the objects from MWT blobs files.
    frame = find_nearest_index(experiment.frame_times, time) + 1        
    bids = experiment.blobs_in_frame(frame)
    #outlines, centroids, outline_ids = [], [], []
    parser = frame_parser_spec(frame)
    blob_data = []
    for bid in bids:
        blob = experiment.parse_blob(bid, parser)
        if blob['contour_encode_len'][0]:
            outline = blob_reader.decode_outline(
                blob['contour_start'][0],
                blob['contour_encode_len'][0],
                blob['contour_encoded'][0],
                )
            blob_data.append((bid, blob['centroid'][0], outline))
    return frame, blob_data


def analyze_image(experiment, time, img, background, threshold, show=False):
    """
    analyze a single image and return results.


    """
    mask = create_binary_mask(img, background, threshold)
    labels, n_img = ndimage.label(mask)
    image_objects = regionprops(labels)
    
    frame, blob_data = grab_blob_data(experiment, time)
    bids, blob_centroids, outlines = zip(*blob_data)
    matches, failures, blobs_to_join, key_outlines, lines = match_objects(bids, blob_centroids, outlines, image_objects)

    # show how well blobs are matched at this threshold.
    if show:
        f, ax = plt.subplots()
        ax.imshow(img.T, cmap=plt.cm.Greys_r)

        ax.contour(mask.T, [0.5], linewidths=1.2, colors='b')
        for outline in outlines:
            ax.plot(*outline.T, color='red')
        for line in lines:
            x, y = line
            ax.plot(x, y, '.-', color='green', lw=2)
        plt.show()

    # compile matches/failures
    compile_all = [(frame, bid, bid in matches) for bid in bids]                
    image_check = pd.DataFrame(compile_all, columns=['frame', 'bid', 'good'])
    image_check['join'] = '' 

    for bs in blobs_to_join:
        join_key = '-'.join([str(i) for i in bs])
        #print(bs, join_key)        
        for b in bs:
            #print(image_check['bid'] == b)
            image_check['join'][image_check['bid'] == b] = join_key

    # order outlines.
    image_outlines = pd.DataFrame(key_outlines, index=['x', 'y', 'l', 'code']).T
    image_outlines['frame'] = frame
    image_outlines.reset_index(inplace=True)
    #print(image_outlines.head(20))
    #print(image_check.head(50))
    return image_check, image_outlines


def analyze_ex_id_images(ex_id, threshold):
    """
    analyze all images for a given ex_id and saves the results to h5 files.

    params
    -------
    ex_id: (str)
        experiment id
    threshold: (float)
        threshold to use when analyzing images.
    """

    # grab images and times.
    times, impaths = grab_images_in_time_range(ex_id, start_time=0)
    times = [float(t) for t in times]
    times, impaths = zip(*sorted(zip(times, impaths)))

    # create recording background
    background = create_backround(impaths)

    # initialize experiment
    path = os.path.join(MWT_DIR, ex_id)
    experiment = multiworm.Experiment(path)
    experiment.load_summary()

    # switch to loop when working
    #i = 6
    #time, impath = times[i], impaths[i]
    #if True:
    full_experiment_check = []
    all_blobs_to_join = []
    all_image_outlines = []

    for i, (time, impath) in enumerate(zip(times, impaths)):
        # get the objects from the image
        #print(impath)
        img = mpimg.imread(impath)
        image_check, image_outlines = analyze_image(experiment, time, img, background, threshold, False)
        full_experiment_check.append(image_check)
        all_image_outlines.append(image_outlines)

        # TODO: remove this failsafe.
        #if i > 3:
        #    break

    # save comprehensive
    image_check = pd.concat(full_experiment_check)    
    image_outlines = pd.concat(all_image_outlines)
    print(image_check)

    ensure_dir_exists(VALIDATION_DIR)
    savename = os.path.join(VALIDATION_DIR, '{eid}.csv'.format(eid=ex_id))
    print(savename)
    image_check.to_csv(savename)
    #rt = pd.read_csv('test.csv', index_col=0)

if __name__ == '__main__':
    ex_id = '20130614_120518'
    ex_id = '20130318_131111'
    threshold = 0.0001
    #threshold = 0.0003
    analyze_ex_id_images(ex_id, threshold)