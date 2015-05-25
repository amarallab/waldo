# This notebook is for exploring methods for finding worms in a recording's images.
# It is intended as an alternative method of validating the MultiWorm Tracker's results.

# standard imports
from __future__ import absolute_import, division, print_function

from six.moves import (zip)

# standard library
from math import fabs

# third party
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# package specific
from waldo.wio import file_manager as fm

from . import manipulations as mim

# Derived from http://stackoverflow.com/a/2566508/194586
# However, I claim these as below the threshold of originality
from .blob_interface import grab_blob_data
from .summarize import match_objects, analyze_image


def draw_colors_on_image(experiment, time, ax=None, colors=None):

    if colors is None:
        c = {'missed_color': 'b',
             'tp_color': 'green',
             'fp_color': 'red',
             'roi_color': 'yellow',
             'roi_line_color': 'blue'}
    else:
        c = colors
    # grab images and times.
    times, impaths = zip(*experiment.image_files)
    #times, impaths = grab_images.grab_images_in_time_range(ex_id, start_time=0)
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

    if roi is not None:
        # draw full circle region of interest
        roi_t = np.linspace(0, 2 * np.pi, 500)
        roi_x = roi['r'] * np.cos(roi_t) + roi['x']
        roi_y = roi['r'] * np.sin(roi_t) + roi['y']
        ax.plot(roi_x, roi_y, color=c['roi_line_color'])
        # resize figure
        ymax, xmax = img.T.shape
        ax.set_xlim([0, xmax])
        ax.set_ylim([0, ymax])
    print('done')
    if show_by_default:
        plt.show()


def draw_colors_on_image_T(experiment, time, ax=None, colors=None):

    if colors is None:
        c = {'missed_color': 'b',
             'tp_color': 'green',
             'fp_color': 'red',
             'roi_color': 'yellow',
             'roi_line_color': 'blue'}
    else:
        c = colors
    # grab images and times.
    times, impaths = zip(*experiment.image_files)
    #times, impaths = grab_images.grab_images_in_time_range(ex_id, start_time=0)
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
    # print(img.shape, h/w, w/h)

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
    # print('outside', objects_outside)

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

    if roi is not None:
        # draw full circle region of interest
        roi_t = np.linspace(0, 2 * np.pi, 500)
        roi_x = roi['r'] * np.cos(roi_t) + roi['x']
        roi_y = roi['r'] * np.sin(roi_t) + roi['y']
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


def draw_minimal_colors_on_image_T(experiment, time, color=None, ax=None):
    # grab images and times.
    times, impaths = zip(*experiment.image_files)
    #times, impaths = grab_images.grab_images_in_time_range(ex_id, start_time=0)
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
    # experiment = wio.Experiment(experiment_id=ex_id)

    time = closest_time
    img = mpimg.imread(closest_image)
    h, w = img.shape
    # print(img.shape, h/w, w/h)
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

    # if roi != None:
    if roi is not None:
        # draw full circle region of interest
        roi_t = np.linspace(0, 2 * np.pi, 500)
        roi_x = roi['r'] * np.cos(roi_t) + roi['x']
        roi_y = roi['r'] * np.sin(roi_t) + roi['y']
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


def show_matched_image(experiment, threshold, time, roi=None):
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
    times, impaths = zip(*experiment.image_files)
    #times, impaths = grab_images.grab_images_in_time_range(ex_id, start_time=0)
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
    #path = paths.experiment(ex_id)
    #experiment = multiworm.Experiment(path)

    time = closest_time
    img = mpimg.imread(closest_image)
    bid_matching, base_acc = analyze_image(experiment, time, img,
                                           background, threshold,
                                           roi=roi, show=True)
    return bid_matching, base_acc


