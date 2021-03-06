from __future__ import absolute_import, division, print_function

import numpy as np
import scipy
import skimage

# import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# from . import grab_images
# from . import draw
import waldo.wio as wio
from . import summarize
from . import manipulations as mim
import waldo.wio.roi_manager as roim


def get_background_and_worm_pixels(background, roi_mask, threshold, impaths):
    # grab all worm shapes from all frames
    n_images = len(impaths)
    print(n_images, 'images')
    if n_images > 10:
        n_images = 10

    worm_values = []
    for imp in impaths[:n_images]:
        img = mpimg.imread(imp)

        mask = mim.create_binary_mask(img, background, threshold)
        # mask = create_binary_mask(img, background, threshold)
        mask = mask * roi_mask
        labels, n_img = scipy.ndimage.label(mask)
        image_objects = skimage.measure.regionprops(labels)

        for o in image_objects:
            bbox = o.bbox
            nmask = o.filled_image
            xmin, ymin, xmax, ymax = bbox
            cut = img[xmin:xmax, ymin:ymax]
            v = list((nmask*cut).flatten())
            worm_values.extend(v)

    background_values = (roi_mask * background).flatten()

    def take_nonzero(a):
        a = np.array(a)
        return a[np.nonzero(a)]

    worm_values = take_nonzero(worm_values)
    background_values = take_nonzero(background_values)
    return worm_values, background_values


def score_images(worm_values, background_values):
    p5 = np.percentile(background_values, 5)
    # print('threshold:', p5, '(5th percentile of background)')

    print('worm_values', worm_values)
    print('p5', p5)
    good_fraction = (worm_values <= p5).sum(dtype=float) / len(worm_values)
    good_fraction = round(good_fraction, ndigits=2)

    contrast_ratio = np.mean(background_values) / np.mean(worm_values)
    contrast_diff = np.mean(background_values) - np.mean(worm_values)
    scores = {'good_fraction': good_fraction,
              'contrast_ratio': contrast_ratio,
              'contrast_diff': contrast_diff}

    return scores


# def make_pixel_histogram(worm_values, background_values, n_bins=100):

#     mmax = max([np.max(worm_values),  np.max(background_values)])
#     bins = np.linspace(0, mmax, n_bins)
#     b = bins[:-1]

#     h0, bin_edges = np.histogram(worm_values, bins=bins)
#     h1, bin_edges = np.histogram(background_values, bins=bins)

#     def norm(a):
#         return a / np.sum(a, dtype=float)

#     fig, ax = plt.subplots()

#     # ax.plot(b, h0, label='worms')
#     # ax.plot(b, h1, label='background')
#     h0 = norm(h0)
#     h1 = norm(h1)

#     # ax.plot(worm_values)
#     ax.plot(b, h0, label='worms1')
#     ax.plot(b, h1, label='background1')

#     ax.legend()
#     ax.set_xlabel('pixel intensity')
#     ax.set_ylabel('p')
#     # plt.show()


def score(ex_id, experiment=None):

    if experiment is None:
        experiment = wio.Experiment(experiment_id=ex_id)

    pfile = wio.file_manager.ImageMarkings(ex_id=experiment.id)
    threshold = pfile.threshold()
    roi_dict = pfile.roi()

    times, impaths = zip(*sorted(experiment.image_files.items()))
    impaths = [str(s) for s in impaths]

    background = mim.create_backround(impaths)

    #x, y, r = roi['x'], roi['y'], roi['r']
    #roi_mask = mim.create_roi_mask(x, y, r, shape=background.shape)
    roi_mask =roim.create_roi_mask(roi_dict, shape=background.shape)

    worm_i, background_i = get_background_and_worm_pixels(background, roi_mask,
                                                          threshold, impaths)
    scores = score_images(worm_i, background_i)
    # print('worm', min(worm_i), np.mean(worm_i), max(worm_i))
    # print('background', min(background_i), np.mean(background_i),
    # max(background_i))
    # print(len(worm_i), len(background_i))
    # make_pixel_histogram(worm_i, background_i)

    img = mpimg.imread(impaths[-1])
    time = times[-1]
    # print(threshold, type(threshold))
    # print(roi, type(roi))
    _, base_acc, _ = summarize.analyze_image(experiment, time, img,
                                             background, threshold,
                                             roi_dict, show=False)
    # print(base_acc)
    false_neg = base_acc['false-neg']
    false_pos = base_acc['false-pos']
    true_pos = base_acc['true-pos']
    accuracy = true_pos / (false_pos + true_pos)
    coverage = true_pos / (true_pos + false_neg)

    scores.update({'accuracy': round(accuracy, ndigits=2),
                   'coverage': round(coverage, ndigits=2)})

    print(scores)
    return scores
