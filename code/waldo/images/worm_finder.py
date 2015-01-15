# This notebook is for exploring methods for finding worms in a recording's images.
# It is intended as an alternative method of validating the MultiWorm Tracker's results.

# standard imports
from __future__ import absolute_import, division, print_function
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import os

from math import fabs
import functools
# third party
import numpy as np
#import scipy
from scipy import ndimage
import pandas as pd
from skimage.measure import regionprops

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# package specific
from waldo import wio
from waldo.wio import paths
from waldo.wio import file_manager as fm

from waldo.extern import multiworm
from multiworm.readers import blob as blob_reader

from . import manipulations as mim
from . import grab_images

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

def grab_blob_data(experiment, time):
    """
    pull the frame number and a list of tuples (bid, centroid, outline)
    for a given time and experiment.

    params
    -------
    experiment: (experiment object from wio)
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
    bad_blobs = []

    for bid in bids:
        try:
            blob = experiment._parse_blob(bid, parser)
            if blob['contour_encode_len'][0]:
                outline = blob_reader.decode_outline(
                    blob['contour_start'][0],
                    blob['contour_encode_len'][0],
                    blob['contour_encoded'][0],
                )
                blob_data.append((bid, blob['centroid'][0], outline))
        except ValueError:
            bad_blobs.append(bid)
    if bad_blobs:
        if len(bad_blobs)> 5:
            print('Warning: {n} blobs failed to load data'.format(n=len(bad_blobs)))
        else:
            print('Warning: {n} blobs failed to load data {ids}'.format(n=len(bad_blobs), ids=bad_blobs))
    return frame, blob_data

def reformat_missing(df):
    """
    reformats missing objects DataFrame to be ready for storage.

    Two main chages:

    - index is changed to be strings beginning with the letter 'm'
    - if an object in the subsequent image is detected in the same
    place as a current object, that object's id is recorded in the
    'next' column'
    - the order of the columns is changed.

    params
    -----
    df: (pandas DataFrame)
        a dataframe containing:
        'f' - (int) frame number
        't' - (float) time
        'x', 'y' - centroid coordinates
        'xmin', 'ymin', 'xmax', 'ymax' -- bounding box coordinates.

    returns
    -----
    df: (pandas DataFrame)
        the reformatted dataframe

    """
    # Reformat names of objects
    #df.reindex(index=range(len(df)))
    #df.reset_index(index=range(len(df)))
    #print(df)
    ids = ['m{i}'.format(i=i) for i in range(len(df.index))]
    frames = list(set(df['f']))
    frames.sort()
    df['id'] = ids
    df = df.set_index('id')

    # find any potential matches
    for i, f in enumerate(frames[:-1]):
        current_df = df[df['f'] == f]
        next_df = df[df['f'] == frames[i+1]]
        matches = {}
        for c_id, c in current_df.iterrows():

            bbox = c['xmin'], c['ymin'], c['xmax'], c['ymax']
            x, y = c['x'], c['y']

            for n_id, n in next_df.iterrows():
                bbox2 = n['xmin'], n['ymin'], n['xmax'], n['ymax']
                nx, ny = n['x'], n['y']
                if mim.do_boxes_overlap(bbox, bbox2):
                    match = matches.get(c_id, None)
                    # save this match if no other match already found.
                    if not match:
                        matches[c_id] = n_id

                    # save closer match if other match found.
                    else:
                        m = df.loc[match]
                        mx, my = m['x'] ,m['y']
                        nd = np.sqrt((x-nx)**2 + (y-ny)**2)
                        md = np.sqrt((x-mx)**2 + (y-my)**2)
                        if nd < md:
                            matches[c_id] = n_id

    print('persisting image objects:')
    print(matches)

    next_list = [matches.get(c_id, ' ') for c_id in df.index]
    df['next'] = next_list

    return df[['f', 't', 'x', 'y',
               'xmin', 'ymin', 'xmax', 'ymax', 'next']]

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
    outside_objects = []

    #print(img_outside_roi)
    if roi != None:
        dx = img_centroids[:, 0] - roi['x']
        dy = img_centroids[:, 1] - roi['y']
        img_roi_check = (np.sqrt(dx** 2 + dy** 2) <= roi['r'])

        for l, in_roi in zip(img_labels, img_roi_check):
            if not in_roi:
                outside_objects.append(l)

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


        is_matched_to_object = False
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

                if mim.do_boxes_overlap(img_bbox, blob_bbox):
                    closest_obj = im_obj
                    closest_cent = im_obj.centroid
                    closest_dist = d

        if closest_obj != -1:
            # for match bid outline must have more overlapping than
            # overreaching pixels.
            # ie. object must be mostly on top of the image_object
            outline_mat = mim.outline_to_outline_matrix(outline,
                                                    bbox=blob_bbox)
            obj_bbox, obj_img = closest_obj.bbox, closest_obj.image
            coord_match = mim.coordiate_match_offset_arrays(blob_bbox,
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
            #underlaps = (overlay == 2).sum()
            overreaches = (overlay == 1).sum()

            # if the objects are mostly on top of one another,
            #count as validated match.
            if overlaps > overreaches:
                # this blob is officially validated.
                is_matched_to_object = True
                matches.append(bid)

                # save for false neg and joins calculations
                blobs_by_object[closest_obj.label].append(bid)

                # save a connecting line for visual validation.
                xs = [cent[0], closest_cent[0]]
                ys = [cent[1], closest_cent[1]]
                lines.append((xs, ys))

        if not is_matched_to_object:
            # this is officially a false positive.
            # no object in image analysis corresponed to it.
            false_pos.append(bid)
            # remove this check when I'm sure it is not happening
            if roi != None:
                dx, dy = (cent[0] - roi['x']), (cent[1] - roi['y'])
                inside_roi = (np.sqrt(dx** 2 + dy** 2) <= roi['r'])
                if not inside_roi:
                    print ('Warning! obj outside roi counted as FP')

    # store locations of missing data
    # loop through all image objects
    missing_data = []
    for im_obj, in_roi in zip(image_objects, img_roi_check):

        matching_blobs = blobs_by_object.get(im_obj.label, [])
        #print(im_obj.label, matching_blobs, in_roi)
        # test if inside roi and has not blob matches.
        if in_roi and not matching_blobs:

            x, y = im_obj.centroid
            xmin, ymin, xmax, ymax = im_obj.bbox
            m = {'x': x, 'y':y,
                 'xmin':xmin, 'ymin':ymin,
                 'xmax':xmax, 'ymax':ymax}
            #print(m)
            missing_data.append(m)
    missing_objects = pd.DataFrame(missing_data)
    blobs_to_join = []              # reformat joins
    false_neg_count = 0                   # initialize missed count

    for label, in_roi in zip(img_labels, img_roi_check):
        matched_ids = blobs_by_object[label]
        if len(matched_ids) > 1:
            blobs_to_join.append(matched_ids)
        if len(matched_ids) == 0 and in_roi:
            false_neg_count += 1

    if verbose:
        print(len(blob_centroids), 'blobs tracked by MWT')
        print(len(false_pos), 'blobs without matches')
        print(len(matches), 'blobs matched to image objects')
        print(len(bid_outside_roi), 'bid outsid roi')
        print(img_outside_roi, 'img outsid roi')

    more = {'blobs_by_object': blobs_by_object,
            'false-neg': false_neg_count,
            'false-pos': len(false_pos),
            'true-pos': len(matches),
            'lines': lines,
            'roi':roi,
            'bid-outside':bid_outside_roi,
            'img-outside':img_outside_roi,
            'outside_objects':outside_objects,
            'missing_df': missing_objects}

    return matches, false_pos, blobs_to_join, more

def analyze_image(experiment, time, img, background, threshold,
                  roi=None, show=True):
    """
    analyze a single image and return results.
    """
    mask = mim.create_binary_mask(img, background, threshold)
    labels, n_img = ndimage.label(mask)
    image_objects = regionprops(labels)

    frame, blob_data = grab_blob_data(experiment, time)
    #print(frame)
    bids, blob_centroids, outlines = zip(*blob_data)
    match = match_objects(bids, blob_centroids, outlines,
                          image_objects, roi=roi)
    matches, false_pos, blobs_to_join, more = match

    ##### for plotting

    # show how well blobs are matched at this threshold.
    if show:
        f, ax = plt.subplots()
        ax.imshow(img.T, cmap=plt.cm.Greys_r)
        ax.contour(mask.T, [0.5], linewidths=1.2, colors='b')
        for outline in outlines:
            ax.plot(*outline.T, color='red')

        lines = more['lines']
        #print(len(lines), 'lines')
        for line in lines:
            x, y = line
            ax.plot(x, y, '.-', color='green', lw=2)

        if roi != None:
            # draw full circle region of interest
            roi_t = np.linspace(0, 2* np.pi, 500)
            roi_x = roi['r'] * np.cos(roi_t) + roi['x']
            roi_y = roi['r'] * np.sin(roi_t)+ roi['y']
            ax.plot(roi_x, roi_y)
            # resize figure
            ymax, xmax = img.T.shape
            ax.set_xlim([0, xmax])
            ax.set_ylim([0, ymax])
        return f, ax


    #### for acuracy calculations

    base_accuracy = {'frame':frame, 'time':time,
                     'false-neg':more['false-neg'],
                     'false-pos':more['false-pos'],
                     'true-pos':more['true-pos']}

    #### for matching blobs to img objects

    # consolidate history of matching objects.
    outside = []
    if roi != None:
        outside = more['bid-outside']

    cols = ['frame', 'bid', 'good', 'roi']
    matching_history = [(frame, bid,
                         bid in matches,
                         bid not in outside)
                        for bid in bids]
    bid_matching = pd.DataFrame(matching_history,
                                columns=cols)
    bid_matching['join'] = ''
    for bs in blobs_to_join:
        join_key = '-'.join([str(i) for i in bs])
        #print(bs, join_key)
        for b in bs:
            #print(bid_matching['bid'] == b)
            bid_matching['join'][bid_matching['bid'] == b] = join_key

    assert more['true-pos'] == len(matches)

    #### for missing img data

    missing_df = more['missing_df']
    missing_df['f'] = int(frame)
    missing_df['t'] = time
    return bid_matching, base_accuracy, missing_df

def draw_colors_on_image(ex_id, time, ax=None, colors=None):

    if colors is None:
        c = {'missed_color': 'b',
             'tp_color':'green',
             'fp_color': 'red',
             'roi_color': 'yellow',
             'roi_line_color': 'blue'}
    else:
        c = colors
    # grab images and times.
    times, impaths = grab_images.grab_images_in_time_range(ex_id, start_time=0)
    times = [float(t) for t in times]
    times, impaths = zip(*sorted(zip(times, impaths)))

    closest_time, closest_image = 1000000.0, None
    for i, (t, impath) in enumerate(zip(times, impaths)):
        if fabs(t - time) < fabs(closest_time - time):
            closest_time = t
            closest_image = impath

    print('looking for {t}'.format(t=time))
    print('closest image is at time {t}'.format(t=closest_time))
    print(closest_image)

    # initialize experiment
    experiment = wio.Experiment(experiment_id=ex_id)

    time = closest_time
    img = mpimg.imread(closest_image)
    h, w = img.shape
    print(img.shape, h/w, w/h)
    prepdata = fm.PrepData(ex_id)
    good = prepdata.good()
    bad = prepdata.bad()
    outside = prepdata.outside()


    frame, blob_data = grab_blob_data(experiment, time)
    print(frame)
    bids, blob_centroids, outlines = zip(*blob_data)

    pf = fm.ImageMarkings(ex_id=ex_id)
    threshold = pf.threshold()
    roi = pf.roi()

    background = mim.create_backround(impaths)
    mask = mim.create_binary_mask(img, background, threshold)
    labels, n_img = ndimage.label(mask)
    image_objects = regionprops(labels)

    match = match_objects(bids, blob_centroids, outlines,
                          image_objects, roi=roi)
    matches, false_pos, blobs_to_join, more = match
    blobs_by_obj = more['blobs_by_object']
    objects_outside = more['outside_objects']
    #print('outside', objects_outside)

    show_by_default = False
    if ax is None:
        show_by_default = True
        f, ax = plt.subplots()
    ax.imshow(img.T, cmap=plt.cm.Greys_r)
    print(len(bids), 'bids found')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    object_baseline = np.zeros(img.shape)
    for o in image_objects:
        if o.label not in blobs_by_obj:
            continue
        if o.label in objects_outside:
            continue
        if len(blobs_by_obj.get(o.label, [])):
            continue
        xmin, ymin, xmax, ymax = o.bbox
        object_baseline[xmin: xmax, ymin: ymax] = o.imag
    ax.contour(object_baseline.T, [0.5], linewidths=1.2,
               colors=c['missed_color'])

    for bid, outline in zip(bids, outlines):
        color = 'blue'
        if bid in good:
            color = c['tp_color']
        if bid in bad:
            color = c['fp_color']
        if bid in outside:
            color = c['roi_color']

        if not len(outline):
            continue
        x, y = zip(*outline)
        ax.fill(x, y, color=color, alpha=0.5)

    if roi != None:
        # draw full circle region of interest
        roi_t = np.linspace(0, 2* np.pi, 500)
        roi_x = roi['r'] * np.cos(roi_t) + roi['x']
        roi_y = roi['r'] * np.sin(roi_t)+ roi['y']
        ax.plot(roi_x, roi_y, color=c['roi_line_color'])
        # resize figure
        ymax, xmax = img.T.shape
        ax.set_xlim([0, xmax])
        ax.set_ylim([0, ymax])
    print('done')
    if show_by_default:
        plt.show()


def draw_colors_on_image_T(ex_id, time, ax=None, colors=None):

    if colors is None:
        c = {'missed_color': 'b',
             'tp_color':'green',
             'fp_color': 'red',
             'roi_color': 'yellow',
             'roi_line_color': 'blue'}
    else:
        c = colors
    # grab images and times.
    times, impaths = grab_images.grab_images_in_time_range(ex_id, start_time=0)
    times = [float(t) for t in times]
    times, impaths = zip(*sorted(zip(times, impaths)))

    closest_time, closest_image = 1000000.0, None
    for i, (t, impath) in enumerate(zip(times, impaths)):
        if fabs(t - time) < fabs(closest_time - time):
            closest_time = t
            closest_image = impath

    print('looking for {t}'.format(t=time))
    print('closest image is at time {t}'.format(t=closest_time))
    print(closest_image)

    # initialize experiment
    experiment = wio.Experiment(experiment_id=ex_id)

    time = closest_time
    img = mpimg.imread(closest_image)
    h, w = img.shape
    #print(img.shape, h/w, w/h)


    frame, blob_data = grab_blob_data(experiment, time)
    print(frame)
    prepdata = fm.PrepData(ex_id)
    good = prepdata.good(frame=frame)
    bad = prepdata.bad(frame=frame)
    outside = prepdata.outside()
    print('good', len(good))
    print('bad', len(bad))
    print('overlap', len(set(good) & set(bad)))


    bids, blob_centroids, outlines = zip(*blob_data)

    pf = fm.ImageMarkings(ex_id=ex_id)
    threshold = pf.threshold()
    roi = pf.roi()

    background = mim.create_backround(impaths)
    mask = mim.create_binary_mask(img, background, threshold)
    labels, n_img = ndimage.label(mask)
    image_objects = regionprops(labels)

    match = match_objects(bids, blob_centroids, outlines,
                          image_objects, roi=roi)
    matches, false_pos, blobs_to_join, more = match
    blobs_by_obj = more['blobs_by_object']
    objects_outside = more['outside_objects']
    #print('outside', objects_outside)

    show_by_default = False
    if ax is None:
        show_by_default = True
        f, ax = plt.subplots()

    ax.imshow(img, cmap=plt.cm.Greys_r)
    print(len(bids), 'bids found')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    object_baseline = np.zeros(img.shape)
    for o in image_objects:
        if o.label not in blobs_by_obj:
            continue
        if o.label in objects_outside:
            continue
        if len(blobs_by_obj.get(o.label, [])):
            continue
        xmin, ymin, xmax, ymax = o.bbox
        object_baseline[xmin: xmax, ymin: ymax] = o.image
    ax.contour(object_baseline, [0.5], linewidths=1.2,
               colors=c['missed_color'])

    for bid, outline in zip(bids, outlines):
        color = 'blue'
        if bid in good:
            color = c['tp_color']
        elif bid in bad:
            color = c['fp_color']
        elif bid in outside:
            color = c['roi_color']

        if not len(outline):
            continue
        x, y = zip(*outline)
        ax.fill(y, x, color=color, alpha=0.5)

    if roi != None:
        # draw full circle region of interest
        roi_t = np.linspace(0, 2* np.pi, 500)
        roi_x = roi['r'] * np.cos(roi_t) + roi['x']
        roi_y = roi['r'] * np.sin(roi_t)+ roi['y']
        ax.plot(roi_y, roi_x, color=c['roi_line_color'])
        # resize figure
        ymax, xmax = img.shape
        print()
        print()
        print(ymax/xmax, 'ymax/xmax')
        print (xmax/ymax, 'xmax/ymax')
        ax.set_xlim([0, xmax])
        ax.set_ylim([0, ymax])
    print('done')
    if show_by_default:
        plt.show()

def draw_minimal_colors_on_image_T(ex_id, time, color=None, ax=None):
    # grab images and times.
    times, impaths = grab_images.grab_images_in_time_range(ex_id, start_time=0)
    times = [float(t) for t in times]
    times, impaths = zip(*sorted(zip(times, impaths)))

    closest_time, closest_image = 1000000.0, None
    for i, (t, impath) in enumerate(zip(times, impaths)):
        if fabs(t - time) < fabs(closest_time - time):
            closest_time = t
            closest_image = impath

    print('looking for {t}'.format(t=time))
    print('closest image is at time {t}'.format(t=closest_time))
    print(closest_image)

    # initialize experiment
    #experiment = wio.Experiment(experiment_id=ex_id)

    time = closest_time
    img = mpimg.imread(closest_image)
    h, w = img.shape
    #print(img.shape, h/w, w/h)
    pf = fm.ImageMarkings(ex_id=ex_id)
    roi = pf.roi()

    show_by_default = False
    if ax is None:
        show_by_default = True
        f, ax = plt.subplots()

    if color is None:
        color = ax._get_lines.color_cycle.next()

    ax.imshow(img, cmap=plt.cm.Greys_r)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    if roi != None:
        # draw full circle region of interest
        roi_t = np.linspace(0, 2* np.pi, 500)
        roi_x = roi['r'] * np.cos(roi_t) + roi['x']
        roi_y = roi['r'] * np.sin(roi_t)+ roi['y']
        ax.plot(roi_y, roi_x, color=color)
        # resize figure
        ymax, xmax = img.shape
        print()
        print()
        print(ymax/xmax, 'ymax/xmax')
        print (xmax/ymax, 'xmax/ymax')
        ax.set_xlim([0, xmax])
        ax.set_ylim([0, ymax])
    print('done')
    if show_by_default:
        plt.show()


def show_matched_image(ex_id, threshold, time, roi=None):
    """
    shows an image in which objects found through image processing
    are matched against the blobs found by the multiworm tracker.

    params
    -------
    ex_id: (str)
        the experiment id
    threshold: (float)
        a threshold value for image processing. if None,
        will automatically check for cached value.
    roi: (dict)
       contains x, y, and r coordinates for a roi. if None,
       no region of interest will be considered during matching,
       and no region of interest will be drawn onto the image.
    """

    # grab images and times.
    times, impaths = grab_images.grab_images_in_time_range(ex_id, start_time=0)
    times = [float(t) for t in times]
    times, impaths = zip(*sorted(zip(times, impaths)))

    closest_time, closest_image = 1000000.0, None
    for i, (t, impath) in enumerate(zip(times, impaths)):
        if fabs(t - time) < fabs(closest_time - time):
            closest_time = t
            closest_image = impath


    print('looking for {t}'.format(t=time))
    print('closest image is at time {t}'.format(t=closest_time))
    print(closest_image)
    # create recording background
    background = mim.create_backround(impaths)

    # initialize experiment
    path = paths.experiment(ex_id)
    experiment = multiworm.Experiment(path)

    time = closest_time
    img = mpimg.imread(closest_image)
    bid_matching, base_acc = analyze_image(experiment, time, img,
                                           background, threshold,
                                           roi=roi, show=True)
    return bid_matching, base_acc

def analyze_ex_id_images(ex_id, threshold, roi=None, callback=None):
    """
    analyze all images for a given ex_id and saves the results to h5 files.

    params
    -------
    ex_id: (str)
        experiment id
    threshold: (float)
        threshold to use when analyzing images.
    """

    if callback:
        CALLBACK_LOAD_FRAC = 0.08
        CALLBACK_LOOP_FRAC = 0.84
        CALLBACK_SAVE_FRAC = 0.08
        def cb_load(p):
            callback(CALLBACK_LOAD_FRAC * p)
        def cb_loop(p):
            callback(CALLBACK_LOAD_FRAC + CALLBACK_LOOP_FRAC * p)
        def cb_save(p):
            callback(CALLBACK_LOAD_FRAC + CALLBACK_LOOP_FRAC +
                    CALLBACK_SAVE_FRAC * p)
    else:
        cb_load = cb_pri = cb_sec = None
    experiment = wio.Experiment(experiment_id=ex_id)

    print('analzying images')
    # grab images and times.
    #times, impaths = grab_images.grab_images_in_time_range(ex_id, start_time=0)
    times, impaths = zip(*sorted(experiment.image_files.items()))
    #times = [float(t) for t in times]
    #times, impaths = zip(*sorted(zip(times, impaths)))

    impaths = [str(s) for s in impaths]
    # create recording background
    background = mim.create_backround(impaths)

    # initialize experiment

    if callback:
        cb_load(1)

    full_experiment_check = []
    accuracy = []
    full_missing = []
    for i, (time, impath) in enumerate(zip(times, impaths)):
        # get the objects from the image
        #print(impath)
        img = mpimg.imread(impath)
        bid_matching, base_acc, miss = analyze_image(experiment, time, img,
                                               background, threshold,
                                               roi=roi, show=False)
        #print(base_acc)
        full_experiment_check.append(bid_matching)
        accuracy.append(base_acc)
        full_missing.append(miss)

        if callback:
            cb_loop(float(i) / len(times))

    bid_matching = pd.concat(full_experiment_check)
    base_accuracy = pd.DataFrame(accuracy)
    missing_worms = pd.concat(full_missing)
    missing_worms = reformat_missing(missing_worms)

    # save datafiles
    prep_data = experiment.prepdata
    prep_data.dump(data_type='matches', dataframe=bid_matching,
                   index=False)
    cb_save(0.33)
    prep_data.dump(data_type='accuracy', dataframe=base_accuracy,
                   index=False)
    cb_save(0.66)
    prep_data.dump(data_type='missing', dataframe=missing_worms,
                   index=True)
    cb_save(1)

def summarize(ex_id, overwrite=True, callback=None):
    """ short script to load threshold, roi and run
    analyze_ex_id_images.
    """

    pfile = fm.ImageMarkings(ex_id=ex_id)
    threshold = pfile.threshold()
    roi = pfile.roi()
    return analyze_ex_id_images(ex_id, threshold, roi, callback=callback)
