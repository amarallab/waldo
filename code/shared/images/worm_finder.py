# This notebook is for exploring methods for finding worms in a recording's images.
# It is intended as an alternative method of validating the MultiWorm Tracker's results.

# standard imports
from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

import os
import sys
import itertools

#import random
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
from math import fabs
from skimage import morphology
from skimage.measure import regionprops
from skimage.filter.rank import entropy

# nonstandard imports
from .manipulations import create_backround, create_binary_mask, show_threshold, outline_to_outline_matrix, align_outline_matricies
from .manipulations import coordiate_match_offset_arrays, do_boxes_overlap, filled_image_to_outline_points
from .grab_images import grab_images_in_time_range
from wio.file_manager import get_good_blobs, get_timeseries, ensure_dir_exists, Preprocess_File

from annotation.image_validation import Validator
import multiworm
from multiworm.readers import blob as blob_reader
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
    except multiworm.core.MWTDataError:
        pass

    # parse the line and return
    blob = blob_reader.parse([line])
    if blob['frame'][0] != frame:
        raise multiworm.core.MWTDataError("Blob line offset failure")
    return blob


def frame_parser_spec(frame):
    return functools.partial(frame_parser, frame=frame)


def match_objects(bids, blob_centroids, blob_outlines, image_objects,
                  roi=None, maxdist=20, verbose=False):
    """



    """
    #print('matching roi', roi)
    # initialize everything.
    img_labels = [r.label for r in image_objects]
    img_centroids = np.array([r.centroid for r in image_objects])
    img_roi_check = np.array([True for r in image_objects])
    #print('len', len(img_roi_check))
    #print('sum', sum(img_roi_check))
    #print('all', all(img_roi_check))
    img_outside_roi = len(img_roi_check) - sum(img_roi_check)
    bid_outside_roi = []
    #print(img_outside_roi)
    if roi != None:
        dx = img_centroids[:, 0] - roi['x']
        dy = img_centroids[:, 1] - roi['y']
        img_roi_check = (np.sqrt(dx** 2 + dy** 2) <= roi['r'])

    #blob_centroids = np.array(blob_centroids)
    matches, false_pos = [], []

    #additional data
    lines = [] # for graphing
    blobs_by_object = {}
    for l in img_labels:
        blobs_by_object[l] = []

    #loop through MWT's blobs.
    for bid, cent, outline in zip(bids, blob_centroids, blob_outlines):
        # skip if no outline. can't match against image objects.
        if not len(outline):
            continue

        # dont bother matching blob if outside roi
        if roi != None:
            dx, dy = (cent[0] - roi['x']), (cent[1] - roi['y'])
            inside_roi = (np.sqrt(dx** 2 + dy** 2) <= roi['r'])
            if not inside_roi:
                bid_outside_roi.append(bid)
                continue

        # remove this cruft if working properly.
        #if isinstance(outline, np.ndarray):
        #    x, y = outline[:,0], outline[:,1]
        #else:

        x, y = zip(*outline)
        blob_bbox = (min(x), min(y), max(x), max(y))

        # calculate distances to all image object centroids.
        dx = img_centroids[:, 0] - cent[0]
        dy = img_centroids[:, 1] - cent[1]
        dists = np.sqrt(dx** 2 + dy** 2)

        # initialize dummy variables and loop over image objects.
        closest_dist = 10 *  maxdist
        closest_obj = -1

        # loop through all image objects
        for im_obj, d, in_roi in zip(image_objects, dists,
                                     img_roi_check):

            # test ifsufficiently close and inside roi.
            if d < maxdist and d < closest_dist and in_roi:
                # now check if bounding boxes overlap.
                # if boxes overlap, store match.
                img_bbox = im_obj.bbox

                if do_boxes_overlap(img_bbox, blob_bbox):
                    closest_obj = im_obj
                    closest_cent = im_obj.centroid
                    closest_dist = d

        if closest_obj != -1:

            # for match bid outline must have more overlapping than
            # overreaching pixels.
            # ie. object must be mostly on top of the image_object
            outline_mat = outline_to_outline_matrix(outline,
                                                    bbox=blob_bbox)
            obj_bbox, obj_img = closest_obj.bbox, closest_obj.image
            coord_match = coordiate_match_offset_arrays(blob_bbox,
                                                        outline_mat,
                                                        obj_bbox,
                                                        obj_img)
            outline_arr, img_arr, bbox = coord_match
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

            # if the objects are mostly on top of one another,
            #ount as validated match.
            if overlaps > overreaches:
                # this blob is officially validated.
                matches.append(bid)

                # save for false neg and joins calculations
                blobs_by_object[closest_obj.label].append(bid)

                # save a connecting line for visual validation.
                xs = [cent[0], closest_cent[0]]
                ys = [cent[1], closest_cent[1]]
                lines.append((xs, ys))

        else:
            # this is officially a bad blob.
            false_pos.append(bid)


    blobs_to_join = []              # reformat joins
    false_neg = 0                   # initialize missed count

    for label, in_roi in zip(img_labels, img_roi_check):
        matched_ids = blobs_by_object[label]
        if len(matched_ids) > 1:
            blobs_to_join.append(matched_ids)
        if len(matched_ids) == 0 and in_roi:
            false_neg += 1

    if verbose:
        print(len(blob_centroids), 'blobs tracked by MWT')
        print(len(false_pos), 'blobs without matches')
        print(len(matches), 'blobs matched to image objects')
        print(len(bid_outside_roi), 'bid outsid roi')
        print(img_outside_roi, 'img outsid roi')

    more = {'blobs_by_object': blobs_by_object,
            'false-neg': false_neg,
            'false-pos': len(false_pos),
            'true-pos': len(matches),
            'lines': lines,
            'roi':roi,
            'bid-outside':len(bid_outside_roi),
            'img-outside':img_outside_roi}


    return matches, false_pos, blobs_to_join, more

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

def analyze_image(experiment, time, img, background, threshold,
                  roi=None, show=False):
    """
    analyze a single image and return results.

    """
    mask = create_binary_mask(img, background, threshold)
    labels, n_img = ndimage.label(mask)
    image_objects = regionprops(labels)

    frame, blob_data = grab_blob_data(experiment, time)
    bids, blob_centroids, outlines = zip(*blob_data)
    match = match_objects(bids, blob_centroids, outlines,
                          image_objects, roi=roi)
    matches, false_neg, blobs_to_join, more = match

    # show how well blobs are matched at this threshold.
    if show:
        f, ax = plt.subplots()
        ax.imshow(img.T, cmap=plt.cm.Greys_r)

        ax.contour(mask.T, [0.5], linewidths=1.2, colors='b')
        for outline in outlines:
            ax.plot(*outline.T, color='red')
        for line in more['lines']:
            x, y = line
            ax.plot(x, y, '.-', color='green', lw=2)
        plt.show()

    base_accuracy = {'frame':frame, 'time':time,
                     'false-neg':more['false-neg'],
                     'false-pos':more['false-pos'],
                     'true-pos':more['true-pos']}

    # consolidate history of matching objects.
    matching_history = [(frame, bid, bid in matches) for bid in bids]
    bid_matching = pd.DataFrame(matching_history,
                               columns=['frame', 'bid', 'good'])
    bid_matching['join'] = ''
    for bs in blobs_to_join:
        join_key = '-'.join([str(i) for i in bs])
        #print(bs, join_key)
        for b in bs:
            #print(bid_matching['bid'] == b)
            bid_matching['join'][bid_matching['bid'] == b] = join_key

    #TODO: if saving the outlines of image objects
    # were added, this would be a good place.
    return bid_matching, base_accuracy

# def binary_outline_to_points(outline_matrix):
#     f = ndimage.morphology.binary_fill_holes(outline_matrix)
#     bigger_shape = (f.shape[0] + 2, f.shape[1] + 2)

#     up = np.zeros(bigger_shape, dtype=bool)
#     mid = np.zeros(bigger_shape, dtype=bool)
#     down = left = right = np.zeros(bigger_shape)

#     mid[1:-1, 1:-1] = f
#     up[1:-1, 2:] = f
#     down[1:-1, :-2] = f
#     left[:-2, 1:-1] = f
#     right[2:, 1:-1] = f

def show_matched_image(ex_id, threshold, time, roi=None):

    # grab images and times.
    times, impaths = grab_images_in_time_range(ex_id, start_time=0)
    times = [float(t) for t in times]
    times, impaths = zip(*sorted(zip(times, impaths)))

    closest_time, closest_image = 1000000.0, None
    for i, (t, impath) in enumerate(zip(times, impaths)):
        if fabs(t - time) < fabs(closest_time - time):
            closest_time = t
            closest_image = impath

    print('closest image is at time {t}'.format(t=closest_time))
    # create recording background
    background = create_backround(impaths)

    # initialize experiment
    path = os.path.join(MWT_DIR, ex_id)
    experiment = multiworm.Experiment(path)
    experiment.load_summary()

    time = closest_time
    img = mpimg.imread(impath)
    bid_matching, base_acc = analyze_image(experiment, time, img,
                                           background, threshold,
                                           roi=roi, show=True)
    return bid_matching, base_acc


def analyze_ex_id_images(ex_id, threshold, roi=None):
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

    full_experiment_check = []
    accuracy = []

    for i, (time, impath) in enumerate(zip(times, impaths)):
        # get the objects from the image
        #print(impath)
        img = mpimg.imread(impath)
        bid_matching, base_acc = analyze_image(experiment, time, img,
                                               background, threshold,
                                               roi=roi, show=False)
        print(base_acc)
        full_experiment_check.append(bid_matching)
        accuracy.append(base_acc)

        # TODO: remove this failsafe.
        #if i > 3:
        #    break


    bid_matching = pd.concat(full_experiment_check)
    base_accuracy = pd.DataFrame(accuracy)

    # save comprehensive
    ensure_dir_exists(VALIDATION_DIR)
    s1 = os.path.join(VALIDATION_DIR,
                      'matching-{eid}.csv'.format(eid=ex_id))
    print(s1)
    bid_matching.to_csv(s1, index=False)

    s2 = os.path.join(VALIDATION_DIR,
                      'check-{eid}.csv'.format(eid=ex_id))
    print(s2)
    base_accuracy.to_csv(s2, index=False)
    return bid_matching, base_accuracy

def main():

    ex_id = '20130614_120518'
    pfile = Preprocess_File(ex_id=ex_id)
    threshold = pfile.threshold()
    roi = pfile.roi()
    print(threshold)
    return analyze_ex_id_images(ex_id, threshold, roi)
