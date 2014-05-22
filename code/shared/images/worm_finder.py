# This notebook is for exploring methods for finding worms in a recording's images.
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
from threshold_picker import create_backround, create_binary_mask, show_threshold
from grab_images import grab_images_in_time_range, crop_image_around_worm
from wio.file_manager import get_good_blobs, get_timeseries
from joining import multiworm
from joining.multiworm.readers import blob as blob_reader
from settings.local import LOGISTICS
MWT_DIR = LOGISTICS['filesystem_data']
print(MWT_DIR)

# Derived from http://stackoverflow.com/a/2566508/194586
# However, I claim these as below the threshold of originality
def find_nearest_index(seq, value):
    return (np.abs(np.array(seq)-value)).argmin()
#
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


def compare_objects(img, background, threshold):    
    #objects, N, mask = find_objects(img, background, threshold)
    mask = create_binary_mask(img, background, threshold)
    labels, N = ndimage.label(mask)
    regions = regionprops(labels)


    cent_stuff = []

    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')

    centroids = []
    for r in regions:
        centroids.append((r.label, r.centroid))
        cent_stuff.append((r.centroid,
                           r.bbox,
                           r.image))

        # draw rectangle around segmented objects
        minr, minc, maxr, maxc = r.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=0.5)
        ax.add_patch(rect)

    plt.show()
    

def match_objects(bids, blob_centroids, image_objects, maxdist=20, verbose=False):
    #regions = image_objects
    img_labels = [r.label for r in image_objects]
    img_centroids = [r.centroid for r in image_objects]

    img_centroids = np.array(img_centroids)
    blob_centroids = np.array(blob_centroids)

    N1 = len(blob_centroids)
    N2 = len(img_centroids)

    potential_matches = []
    unmatched_blobs = []
    lines = []
    for bid, c1 in zip(bids, blob_centroids):
        d = (img_centroids - c1)
        dists = np.sqrt(d[:,0] ** 2 + d[:,1] ** 2)
        closest_dist = 10 *  maxdist
        closest_obj = -1
        for i, d in enumerate(dists):
            if d < maxdist and d < closest_dist:
                closest_obj = i
                closest_dist = d
                
        if closest_obj != -1:
            potential_matches.append((bid, closest_obj, closest_dist))
            xs = [c1[0], img_centroids[closest_obj][0]]
            ys = [c1[1], img_centroids[closest_obj][1]]
            lines.append((xs, ys))
        else:
            unmatched_blobs.append(bid)

    # TODO: fix this potential bug and resolve conflicts.
    if len(potential_matches):
        a, b, c = zip(*potential_matches)
        if len(b) > len(set(b)):
            print('WARNING: image object matched to more than one blob!')




    if verbose:
        print(N1, 'blobs tracked by MWT')
        print(len(unmatched_blobs), 'blobs without matches')
        print(len(potential_matches), 'blobs matched to image objects')
    return potential_matches, unmatched_blobs, lines

def check_accuracy(img, matches, outline_ids, outlines, image_objects, show=False, verbose=False):

    corrections = {'joins':[], 'splits':[]} 
    bids, true_area, overlaps, underlaps, overreaches = [], [], [], [], []
    regions = image_objects
    metrics = []

    for (bid, oid, _) in matches:

        # initialize data in the coordinate system of the image object.
        outline = outlines[outline_ids.index(bid)]
        x, y = zip(*outline)
        ominx, ominy, omaxx, omaxy = min(x), min(y), max(x), max(y)
        
        r = regions[oid]
        minc, minr, maxc, maxr = r.bbox
        obj_img = r.image.T

        minx = int(min([ominx, minc]))
        miny = int(min([ominy, minr]))
        maxx = int(max([omaxx, maxc]))
        maxy = int(max([omaxy, maxr]))



        box_shape = (maxy - miny + 1, maxx - minx + 1)
        slack = np.array(box_shape) - np.array(obj_img.shape)

        # here xy weirdness starts because origional
        xoff1 = minr - miny
        yoff1 = minc - minx
        xoff2 = slack[0] - xoff1
        yoff2 = slack[1] - yoff1

        x = [i - minx for i in x]
        y = [i - miny for i in y]
        a1 = np.zeros(box_shape)

        if verbose:
            print('\n\nid', bid)
            print('bbox', minc, minr, maxc, maxr)
            print('outline', ominx, ominy, omaxx, omaxy)
            print('new', minx, miny, maxx, maxy)
            print('shape', box_shape, obj_img.shape)
            print('slack', slack)
            print('offsets x', xoff1, xoff2)
            print('offsets y', yoff1, yoff2)
            print('shape match?', a1[xoff1:-xoff2, yoff1:-yoff2].shape, obj_img.shape)

        if False:
            # show box around outline.
            fig, ax = plt.subplots()
            ax.imshow(img.T, cmap=plt.cm.Greys_r)
            ax.plot(*outline.T, color='b')
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='green', linewidth=0.5)
            ax.add_patch(rect)
            plt.show()

        # merge matricies so we can see which pixels are from one the other or both.        
        a1[xoff1: - xoff2, yoff1: -yoff2] = 1
        a1[xoff1: - xoff2, yoff1: -yoff2] = a1[xoff1: - xoff2, yoff1: -yoff2] * 2 * obj_img

        outline_matrix = np.zeros(box_shape)
        for i, j in zip(x, y):
            outline_matrix[j, i] = 1
        outline_matrix = ndimage.morphology.binary_fill_holes(outline_matrix)

        a2 = np.ones(box_shape) * outline_matrix
        merge = a1 + a2
    
        #print(bid, r.area, (a1 ==2).sum())

        # save everything
        area = r.area
        true_area.append(area)
        overlaps.append((merge == 3).sum() / float(area))
        underlaps.append((merge == 2).sum() / float(area))
        overreaches.append((merge == 1).sum() / float(area))                           
        metrics = (bids, true_area, overlaps, underlaps, overreaches)

        if False:
            # shapes in seperate coordinate system
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(obj_img)
            ax[0].plot(x, y)
            ax[1].imshow(outline_matrix)
            plt.show()

        if False:
            # show merge
            print('overlap', overlaps[-1])
            print('underlap', underlaps[-1])
            print('overreach', overreaches[-1])

            fig, ax = plt.subplots(1,3)
            ax[0].imshow(a1)
            ax[1].imshow(a2)
            ax[2].imshow(a1 + a2)
        plt.show()

    return metrics, corrections
    
def check_entropy(img):                     
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    ax[0, 0].imshow(img, cmap=plt.cm.Greys_r)
    ax[0, 1].imshow(entropy(img, morphology.disk(3)))
    ax[1, 0].imshow(entropy(img, morphology.disk(5)))
    ax[1, 1].imshow(entropy(img, morphology.disk(20)))

def flip_bbox_xy(bbox):
    return (bbox[1], bbox[0], bbox[3], bbox[2])

def main():
    ex_id = '20130614_120518'
    ex_id = '20130318_131111'
    threshold = 0.0001
    #threshold = 0.0003

    # grab images and times.
    times, impaths = grab_images_in_time_range(ex_id, start_time=0)
    times = [float(t) for t in times]
    times, impaths = zip(*sorted(zip(times, impaths)))
    background = create_backround(impaths)

    # find appropriate threshold for recording.
    mid = mpimg.imread(impaths[int(len(impaths)/2)])
    #threshold = pick_threshold_in_range(img=mid, background=background)
    #show_threshold_spread(mid, background)
    show_threshold(mid, background, threshold)
    #check_entropy(mid)
    plt.show()

    # initialize experiment
    path = os.path.join(MWT_DIR, ex_id)
    experiment = multiworm.Experiment(path)
    experiment.load_summary()


    # initialize containers.        
    metrics = {'coverage':[], 'N-img':[], 'N-mwt':[], 'time':[],
               'matched':[], 'unmatched':[], 'true_area':[],
               'overlap': [], 'underlap': [], 'overreach': []}

    # switch to loop when working

    i = 6
    time, impath = times[i], impaths[i]
    #for i, (time, impath) in enumerate(zip(times, impaths)):
    if True:
        # get the objects from the image
        #print(impath)
        img = mpimg.imread(impath)
        mask = create_binary_mask(img, background, threshold)
        labels, n_img = ndimage.label(mask)
        image_objects = regionprops(labels)

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

        bids, blob_centroids, outlines = zip(*blob_data)
        matches, unmatches, lines = match_objects(bids, blob_centroids, image_objects)

        # TODO: remove spurious image_objects
        good_objects = image_objects

        #print(t, len(good_objects), n_img)

        # save basic tracking numbers
        metrics['time'].append(time)
        metrics['N-img'].append(n_img)
        metrics['N-mwt'].append(len(bids))
        metrics['coverage'].append(float(len(matches)) / float(n_img))
        metrics['matched'].append(len(matches))
        metrics['unmatched'].append(len(unmatches))                
        # accuracy
        acuracy_metrics, corections = check_accuracy(img, matches, bids, outlines, image_objects)
        bids, true_area, overlaps, underlaps, overreaches = acuracy_metrics
        metrics['true_area'].append(overlaps)
        metrics['overlap'].append(np.mean(overlaps))
        metrics['underlap'].append(np.mean(underlaps))
        metrics['overreach'].append(np.mean(overreaches))

        # show how well blobs are matched at this threshold.
        if True:
            f, ax = plt.subplots()
            ax.imshow(img.T, cmap=plt.cm.Greys_r)
            ax.contour(mask.T, [0.5], linewidths=1.2, colors='b')
            for outline in outlines:
                ax.plot(*outline.T, color='red')
            for line in lines:
                x, y = line
                ax.plot(x, y, '.-', color='green', lw=2)
            plt.show()

    if False:
        time = metrics['time']
        #print(time)
        #print(metrics['N-img'])

        f, ax = plt.subplots(4, 1, sharex=True)
        ppl.plot(ax[0], time, metrics['N-img'], '.--', label='N-img')
        ppl.plot(ax[0], time, metrics['N-mwt'], '.--', label='N-mwt')
        ax[0].legend()

        ppl.plot(ax[1], time, metrics['coverage'], label='coverage')
        ax[1].set_ylabel('coverage')

        ppl.plot(ax[2], time, metrics['overlap'], label='TP | overlap')
        ppl.plot(ax[2], time, metrics['underlap'], label='FN | underlap')
        ppl.plot(ax[2], time, metrics['overreach'], label='FP | overreach')

        ax[2].set_ylabel('accuracy')
        ax[2].legend()
        ax[-1].set_xlabel('time')
        ax[-1].set_xlim([time[0], time[-1]])
        plt.show()

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
'''
def join_offset_arrays(corner1, array1, corner2, array2):

    print(array1.shape)
    print(array2.shape)

    # initialize everything
    corner1, corner2 = np.array(corner1), np.array(corner2)
    xmin1, ymin1 = corner1
    xmin2, ymin2 = corner2
    xmax1, ymax1 = corner1 + np.array(array1.shape)
    xmax2, ymax2 = corner2 + np.array(array2.shape)
            
    # create new bounding box
    xmin = int(min([xmin1, xmin2]))
    ymin = int(min([ymin1, ymin2]))
    xmax = int(max([xmax1, xmax2]))
    ymax = int(max([ymax1, ymax2]))


    print(xmin1, ymin1, ymax1, ymax1)
    print(xmin2, ymin2, ymax2, ymax2)
    print(xmin, ymin, ymax, ymax)


    # initialize new array shapes
    box_shape = (xmax - xmin, ymax - ymin)
    new1 = np.zeros(box_shape, dtype=int)
    new2 = np.zeros(box_shape, dtype=int)

    # calculate first offsets.
    yoff1 = ymin1 - ymin
    xoff1 = xmin1 - xmin

    slack1 = np.array(box_shape) - np.array(array1.shape)
    xoff1B = slack1[0] - xoff1
    yoff1B = slack1[1] - yoff1
    new1[xoff1:-xoff1B, yoff1:-yoff1B] = 1

    print(1)
    print('yoff', yoff1, yoff1B)
    print('xoff', xoff1, xoff1B)
    print(slack1) 
    print(new1)

    yoff2 = ymin2 - ymin
    xoff2 = xmin2 - xmin

    slack2 = np.array(box_shape) - np.array(array2.shape)
    print(box_shape, array2.shape)
    xoff2B = slack2[0] - xoff2
    yoff2B = slack2[1] - yoff2

    if xoff2B == 0 and yoff2B == 0:
        print('both')
        new2[xoff2:, yoff2:] = 1
    elif xoff2B == 0:
        print('just x')
        new2[xoff2:, yoff2:-yoff2B] = 1
    elif yoff2B == 0:
        print('just y')
        new2[xoff2:-xoff2, yoff2:] = 1
    else:
        new2[xoff2:-xoff2B, yoff2:-yoff2B] = 1

    print(2)
    print('yoff', yoff2, yoff2B)
    print('xoff', xoff2, xoff2B)
    print(slack2) 
    print(new2)

    if True:
        fig, ax = plt.subplots()
        rect1 = mpatches.Rectangle((xmin1, ymin1), xmax1 - xmin1, ymax1 - ymin1,
                                   fill=True, alpha=0.3, color='red', linewidth=0.5)
        rect2 = mpatches.Rectangle((xmin2, ymin2), xmax2 - xmin2, ymax2 - ymin2,
                                   fill=True, alpha=0.3, color='blue', linewidth=0.5)
        rect3 = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=True, alpha=0.3, color='green', linewidth=0.5)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 20])
        #plt.show()

    


    fig, ax = plt.subplots(1,3)
    ax[0].imshow(new1, cmap=plt.cm.Greys_r)
    ax[1].imshow(new2, cmap=plt.cm.Greys_r)
    ax[2].imshow(new1 + new2, cmap=plt.cm.Greys_r)
    plt.show()
'''

def join_offset_arrays(bbox1, array1, bbox2, array2):
    # calculate new bounding box
    mins = np.minimum(np.array(bbox1), np.array(bbox2))[:2]
    maxs = np.maximum(np.array(bbox1), np.array(bbox2))[2:]
    bbox = (mins[0], mins[1], maxs[0], maxs[1])
    xmin, ymin, xmax, ymax = bbox
    print(bbox)
    

    # initialize everything
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    '''

    # create new bounding box
    xmin = int(min([xmin1, xmin2]))
    ymin = int(min([ymin1, ymin2]))
    xmax = int(max([xmax1, xmax2]))
    ymax = int(max([ymax1, ymax2]))
    print(xmin1, ymin1, ymax1, ymax1)
    print(xmin2, ymin2, ymax2, ymax2)
    print(xmin, ymin, ymax, ymax)
    '''

    # initialize new array shapes
    box_shape = (xmax - xmin, ymax - ymin)
    print('shape', box_shape)

    # TODO once validated remove this
    '''
    def fit_old_array_into_new_shape2(a, off, shape):
        new = np.zeros(shape, dtype=int)
        xoff, yoff = off
        xoff2, yoff2 = off + np.array(a.shape)
        new[xoff:xoff2, yoff:yoff2] = 1
        slack = np.array(shape) - np.array(a.shape)        
        xoff2, yoff2 = slack - np.array(off)
        print(shape)
        print(off + np.array(a.shape))


        if not xoff2 and not yoff2:
            print('both')
            new[xoff:, yoff:] = 1
        elif not xoff2:
            print('just x')
            new[xoff2:, yoff:-yoff2] = 1
        elif not yoff2:
            print('just y')
            new[xoff:-xoff2, yoff:] = 1            
        else:
            new[xoff:-xoff2, yoff:-yoff2] = 1
        return new
    '''
    def fit_old_array_into_new_shape(a, off, shape):
        new = np.zeros(shape, dtype=int)
        xoff, yoff = off
        xoff2, yoff2 = off + np.array(a.shape)
        new[xoff:xoff2, yoff:yoff2] = 1
        return new

    # calculate first offsets.    
    offsets = ((xmin1 - xmin), (ymin1 - ymin))
    new1 = fit_old_array_into_new_shape(a=array1, off=offsets, shape=box_shape)

    offsets = ((xmin2 - xmin), (ymin2 - ymin))
    new2 = fit_old_array_into_new_shape(a=array2, off=offsets, shape=box_shape)
    return new1, new2

if __name__ == '__main__':
    #b1 = (0, 0, 10, 10)
    #b2 = (5, 5, 15, 15)    
    #overlap = do_boxes_overlap(b1, b2)
    '''
    c1 = (5, 5, 10, 11)
    c2 = (6, 7, 12, 15)
    a1 = np.ones((c1[2] - c1[0], c1[3] - c1[1]))
    a2 = np.ones((c2[2] - c2[0], c2[3] - c2[1]))

    join_offset_arrays2(c1, a1, c2, a2)
    '''
    main()
