from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import regionprops

from waldo.images import manipulations as mim
from waldo.images.blob_interface import grab_blob_data
from waldo.wio import file_manager as fm
import waldo.wio.roi_manager as roim

__author__ = 'peterwinter'


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
    #print(ids, 'ids to work with')
    frames = list(set(df['f']))
    frames.sort()
    df['id'] = ids
    df = df.set_index('id')

    # find any potential matches
    matches = {} # dict of all matches
    for i, f in enumerate(frames[:-1]):
        current_df = df[df['f'] == f]
        next_df = df[df['f'] == frames[i+1]]

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
                        nd = (x-nx)**2 + (y-ny)**2
                        md = (x-mx)**2 + (y-my)**2
                        if nd < md:
                            matches[c_id] = n_id

    print(len(matches), 'persisting image objects')

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

    if roi is not None:
        roi_mask = roim.create_roi_mask(roi)
        xs = img_centroids[:, 0]
        ys = img_centroids[:, 1]
        img_roi_check = roim.are_points_inside_mask(xs, ys, roi_mask)
        for l, in_roi in zip(img_labels, img_roi_check):
            if not in_roi:
                outside_objects.append(l)

    # blob_centroids = np.array(blob_centroids)
    matches, false_pos = [], []

    # additional data
    lines = []  # for graphing
    blobs_by_object = {}
    for l in img_labels:
        blobs_by_object[l] = []

    # loop through MWT's blobs.
    for bid, cent, outline in zip(bids, blob_centroids, blob_outlines):
        # skip if no outline. can't match against image objects.
        if not len(outline):
            continue

        # dont bother matching blob if outside roi
        if roi is not None:
            inside_roi = roim.are_points_inside_mask([cent[0]], [cent[1]], roi_mask)
            if not inside_roi:
                bid_outside_roi.append(bid)
                continue

        is_matched_to_object = False
        x, y = zip(*outline)
        blob_bbox = (min(x), min(y), max(x), max(y))

        # calculate distances to all image object centroids.
        dx = img_centroids[:, 0] - cent[0]
        dy = img_centroids[:, 1] - cent[1]
        dists = np.sqrt(dx ** 2 + dy ** 2)

        # initialize dummy variables and loop over image objects.
        closest_dist = 10 * maxdist
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
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(outline_arr)
                ax[1].imshow(img_arr)
                ax[2].imshow(overlay)
                plt.show()

            # calculate pixel matches.
            overlaps = (overlay == 3).sum()
            # underlaps = (overlay == 2).sum()
            overreaches = (overlay == 1).sum()

            # if the objects are mostly on top of one another,
            # count as validated match.
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
            if roi is not None:
                inside_roi = roim.are_points_inside_mask([cent[0]], [cent[1]], roi_mask)
                if not inside_roi:
                    print ('Warning! obj outside roi counted as FP')

    # store locations of missing data
    # loop through all image objects
    missing_data = []
    for im_obj, in_roi in zip(image_objects, img_roi_check):

        matching_blobs = blobs_by_object.get(im_obj.label, [])
        # print(im_obj.label, matching_blobs, in_roi)
        # test if inside roi and has not blob matches.
        if in_roi and not matching_blobs:

            x, y = im_obj.centroid
            xmin, ymin, xmax, ymax = im_obj.bbox
            m = {'x': x, 'y': y,
                 'xmin': xmin, 'ymin': ymin,
                 'xmax': xmax, 'ymax': ymax}
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
            'roi': roi,
            'bid-outside': bid_outside_roi,
            'img-outside': img_outside_roi,
            'outside_objects': outside_objects,
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
    image_objects_exist = len(list(image_objects)) > 0

    frame, blob_data = grab_blob_data(experiment, time)
    # print(frame)
    if len(blob_data) and image_objects_exist:
        bids, blob_centroids, outlines = zip(*blob_data)
        match = match_objects(bids, blob_centroids, outlines,
                              image_objects, roi=roi)
        matches, false_pos, blobs_to_join, more = match
    elif show:
        # no blob data... but make sure plotting succeeds
        outlines = []
        lines = []
    else:
        # no blobs found for frame... return None
        return None, None, None
    # for plotting

    # show how well blobs are matched at this threshold.
    if show:
        f, ax = plt.subplots()
        ax.imshow(img.T, cmap=plt.cm.Greys_r)
        ax.contour(mask.T, [0.5], linewidths=1.2, colors='b')
        for outline in outlines:
            ax.plot(*outline.T, color='red')

        lines = more['lines']
        # print(len(lines), 'lines')
        for line in lines:
            x, y = line
            ax.plot(x, y, '.-', color='green', lw=2)

        if roi is not None:
            # draw full circle region of interest
            roi_x, roi_y = roim.roi_dict_to_points(roi)
            ax.plot(roi_x, roi_y)

            # resize figure
            # TODO: check if this is correct: img.T.shape
            ymax, xmax = img.shape
            ax.set_xlim([0, xmax])
            ax.set_ylim([0, ymax])
        return f, ax

    # for acuracy calculations

    base_accuracy = {'frame': frame, 'time': time,
                     'false-neg': more['false-neg'],
                     'false-pos': more['false-pos'],
                     'true-pos': more['true-pos']}

    # for matching blobs to img objects

    # consolidate history of matching objects.
    outside = []
    if roi is not None:
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
        # print(bs, join_key)
        for b in bs:
            # print(bid_matching['bid'] == b)
            bid_matching['join'][bid_matching['bid'] == b] = join_key

    assert more['true-pos'] == len(matches)

    # for missing img data

    missing_df = more['missing_df']
    missing_df['f'] = int(frame)
    missing_df['t'] = time
    return bid_matching, base_accuracy, missing_df


def analyze_experiment_images(experiment, threshold, roi=None, callback=None,
                              image_callback=None):
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
        cb_load = cb_loop = cb_save = None

    print('analzying images')
    # grab images and times.
    # times, impaths = grab_images.grab_images_in_time_range(ex_id,
    # start_time=0)
    times, impaths = zip(*sorted(experiment.image_files.items()))
    # times = [float(t) for t in times]
    # times, impaths = zip(*sorted(zip(times, impaths)))


    max_images_used = 1000
    if len(times) > max_images_used:
        print('too many images. to save speed waldo will only process 1000 or less')
        times_to_long = int(len(times) / max_images_used)
        times = times[::times_to_long]
        impaths = impaths[::times_to_long]


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
        # print(i, impath)
        img = mpimg.imread(impath)
        if image_callback:
            image_callback(img)
        bid_matching, base_acc, miss = analyze_image(experiment, time, img,
                                                     background, threshold,
                                                     roi=roi, show=False)
        if bid_matching is None or base_acc is None or miss is None:
            continue
        # print(base_acc)
        if full_experiment_check is not None:
            full_experiment_check.append(bid_matching)
        if accuracy is not None:
            accuracy.append(base_acc)

        full_missing.append(miss)

        if callback:
            cb_loop(float(i) / len(times))

    # save datafiles
    prep_data = experiment.prepdata

    if len(full_experiment_check):
        bid_matching = pd.concat(full_experiment_check)
        prep_data.dump(data_type='matches', dataframe=bid_matching,
                       index=False)

    cb_save(0.33)

    if len(accuracy):
        base_accuracy = pd.DataFrame(accuracy)
        prep_data.dump(data_type='accuracy', dataframe=base_accuracy,
                       index=False)

    cb_save(0.66)

    # if there are missing worms, save those too
    if len(accuracy):
        missing_worms = pd.concat(full_missing)
        if len(missing_worms):
            print(len(missing_worms), 'missing blobs found')
            missing_worms = reformat_missing(missing_worms)

        prep_data.dump(data_type='missing', dataframe=missing_worms,
                       index=True)
    cb_save(1)


def summarize_experiment(experiment, callback=None, image_callback=None):
    """ short script to load threshold, roi and run
    analyze_ex_id_images.
    """
    ex_id = experiment.id
    pfile = fm.ImageMarkings(ex_id=ex_id)
    threshold = pfile.threshold()
    roi = pfile.roi()
    return analyze_experiment_images(experiment, threshold, roi, callback=callback,
                                     image_callback=image_callback)
