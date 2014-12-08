# This notebook is for exploring methods for finding worms in a recording's images.
# It is intended as an alternative method of validating the MultiWorm Tracker's results.

# standard imports
from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import os

# third party
import numpy as np
import scipy
from scipy import ndimage
import pandas as pd
from skimage.measure import regionprops

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from .grab_images import grab_images_in_time_range
from .manipulations import create_backround, create_binary_mask

from waldo.wio.experiment import Experiment
import waldo.wio.file_manager as fm
from .worm_finder import grab_blob_data, match_objects

# for IdTracker VALIDATION
def cutouts_for_worms(ex_id, savedir, worm_component_dict):
    """
    """
    worm_dirs = {}
    worms_by_blob = {}
    for worm in worm_component_dict:
        w_dir = '{p}/{eid}/worm_{w}'.format(p=savedir, eid=ex_id,
                                            w=worm)
        w_dir = os.path.abspath(w_dir)
        fm.ensure_dir_exists(w_dir)
        worm_dirs[worm] = w_dir
        for blob in worm_component_dict[worm]:
            worms_by_blob[blob] = worm

    pfile = fm.ImageMarkings(ex_id=ex_id)
    threshold = pfile.threshold()
    roi = pfile.roi()
    print(threshold)
    times, impaths = grab_images_in_time_range(ex_id, start_time=0)
    impaths.sort()
    for i in impaths:
        print(i)
    background = create_backround(impaths)
    full_index = []

    # for blob matching
    experiment = Experiment(experiment_id=ex_id)

    print(len(impaths), 'images')
    for i, (time, impath) in enumerate(zip(times, impaths)):
        img = mpimg.imread(impath)
        mask = create_binary_mask(img, background, threshold)
        labs, n_img = ndimage.label(mask)
        image_objects = regionprops(labs)
        cutouts = cutouts_from_image(img, image_objects, background, roi, threshold)
        labels, masks, imgs, img_index = cutouts
        img_index['time'] = time
        frame, blob_data = grab_blob_data(experiment, float(time))
        img_index['frame'] = frame
        bids, blob_centroids, outlines = zip(*blob_data)
        _, _, _, more = match_objects(bids, blob_centroids, outlines,
                                      image_objects,roi=roi)

        blobs =[]
        worms = []
        for l in img_index['label']:
            match = more['blobs_by_object'][l]
            a = ''
            worm = ''
            if match:
                a = match[0]
                worm = worms_by_blob.get(a, '')
            blobs.append(a)
            worms.append(worm)
        img_index['bid'] = blobs
        img_index['worm'] = worms
        img_index = img_index[img_index['worm'] != '']
        #print(img_index.head(10))

        for l, m, im in zip(labels, masks, imgs):
            bid_matches = more['blobs_by_object'][l]
            if not bid_matches:
                continue
            bid = bid_matches[0]

            worm = worms_by_blob.get(bid, None)
            if worm is None:
                continue

            w_dir = worm_dirs[worm]
            mfile = '{frame}_mask.png'.format(frame=frame)
            ifile = '{frame}_img.png'.format(frame=frame)
            scipy.misc.imsave(os.path.join(w_dir, ifile), im)
            scipy.misc.imsave(os.path.join(w_dir, mfile), m)

        full_index.append(img_index)
        # if i > 2:
        #     break

    index = pd.concat(full_index)
    print(index.head())
    index.to_csv(os.path.join(savedir, 'index.csv'), index=False)

def worm_cutouts(ex_id, savedir, threshold=None, roi=None):
    """
    creates a nested directory structure containing cropped worm images,
    masks corresponding to the worm images, and an index csv with
    relevant information.

    params
    -------
    ex_id: (str)
        the experiment id
    savedir: (str)
        the path in which the nested directory structure will be created.
    threshold: (float)
        a threshold value for image processing. if None,
        will automatically check for cached value.
    roi: (dict)
       contains x, y, and r coordinates for a roi. if None,
       will automatically check for cached value.
    """

    if not threshold or not roi:
        pfile = fm.ImageMarkings(ex_id=ex_id)
        if not threshold:
            threshold = pfile.threshold()
        if not roi:
            roi = pfile.roi()

    times, impaths = grab_images_in_time_range(ex_id, start_time=0)
    background = create_backround(impaths)
    full_index = []

    # for blob matching
    path = os.path.join(settings.MWT_DATA_ROOT, ex_id)
    experiment = multiworm.Experiment(path)

    print(len(impaths), 'images')
    for i, (time, impath) in enumerate(zip(times, impaths)):
        img = mpimg.imread(impath)
        mask = create_binary_mask(img, background, threshold)
        labels, n_img = ndimage.label(mask)
        image_objects = regionprops(labels)
        cutouts = cutouts_from_image(img, image_objects, roi)
        labels, masks, imgs, img_index = cutouts

        img_index['time'] = time
        frame, blob_data = grab_blob_data(experiment, float(time))
        img_index['frame'] = frame
        bids, blob_centroids, outlines = zip(*blob_data)
        _, _, _, more = match_objects(bids, blob_centroids, outlines,
                                      image_objects,roi=roi)

        bbo =[]
        for l in img_index['label']:
            match = more['blobs_by_object'][l]
            a = ''
            if match:
                a = match[0]
            bbo.append(a)
        img_index['bid'] = bbo
        #print(img_index.head(10))

        f_dir = os.path.join(savedir, str(frame))
        fm.ensure_dir_exists(f_dir)
        for l, m, im in zip(labels, masks, imgs):
            bid_matches = more['blobs_by_object'][l]
            if not bid_matches:
                continue
            bid = bid_matches[0]
            mfile = '{bid}_mask.png'.format(bid=bid)
            ifile = '{bid}_i<mg.png'.format(bid=bid)
            scipy.misc.imsave(os.path.join(f_dir, ifile), im)
            scipy.misc.imsave(os.path.join(f_dir, mfile), m)

        full_index.append(img_index)
        #if i > 4:
        #    break

    index = pd.concat(full_index)
    print(index.head())
    index.to_csv(os.path.join(savedir, 'index.csv'), index=False)

def cutouts_from_image(img, image_objects, background, roi, threshold):
    """
    returns a list of images cropped around a worm,
    a list of masks that correspond to each worm image,
    and a pandas DataFrame containing information on each worm
    object.

    params
    -------
    img: (np array)
        contains image data
    image_objects: (scikit.measure.regionprops object)
        contains an iterator for a set of region objects
    roi: (dict)
       contains x, y, and r coordinates for a roi
    """
    labels = []
    masks = []
    imgs = []
    image_index = []
    for obj in image_objects:
        if roi != None:
            x, y = obj.centroid
            dx = x - roi['x']
            dy = y - roi['y']
            # skip if outside roi
            if np.sqrt(dx** 2 + dy** 2) > roi['r']:
                continue

        obj_bbox = obj.bbox
        xmin, ymin, xmax, ymax = obj_bbox
        obj_mask = obj.image
        obj_img = img[xmin:xmax, ymin:ymax]

        labels.append(obj.label)
        imgs.append(obj_img)
        masks.append(obj_mask)
        image_index.append({'label':obj.label,
                            'xmin':xmin, 'xmax':xmax,
                            'ymin':ymin, 'ymax':ymax,
                            'x':x, 'y':y})

        if False:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(obj_mask)
            ax[1].imshow(obj_img, cmap=plt.cm.Greys_r)
            plt.show()
    #mask = create_binary_mask(img, background, threshold)
    #labels, n_img = ndimage.label(mask)
    #image_objects = regionprops(labels)
    return labels, masks, imgs, pd.DataFrame(image_index)
