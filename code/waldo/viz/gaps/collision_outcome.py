# -*- coding: utf-8 -*-
"""
MWT collision visualization (for screening)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import zip, range

import math

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageChops

from waldo.viz.tools import get_contour, patch_contours, Box

def show_collision_choices(experiment, parents, children):
    boxes = []
    blobs = []
    shapes = []
    times = []
    centroids = []

    # load up all the relevant data
    for bid, end in zip([lost_bid, found_bid], ['last', 'first']):
        blob = experiment[bid]
        blob.df.decode_contour()
        blob_shape = get_contour(blob, end)
        blob_box = Box.fit(blob_shape)
        idx = -1 if end == 'last' else 0
        time = blob.df['time'].iloc[idx]
        centroid = blob.df['centroid'].iloc[idx]
        if len(blob_shape) == 1:
            blob_box.size = 30, 30

        blobs.append(blob)
        boxes.append(blob_box)
        shapes.append(blob_shape)
        times.append(time)
        centroids.append(centroid)

    # tweak bounds to make uniformly square images
    bounds = sum(boxes)
    bounds.grow(40)
    min_sq_dim = 150
    if bounds.width < min_sq_dim:
        bounds.width = min_sq_dim
    if bounds.height < min_sq_dim:
        bounds.height = min_sq_dim
    bounds.square()

    # load all images and crop
    image_fns = experiment.image_files.spanning(times=times)
    images = []
    crop_box = Box([int(x) for x in bounds])
    for f in image_fns:
        im, extents = pil_crop(pillowed(str(f)), crop_box)
        images.append(adjust_image(im))

    # merge and convert to array (matplotlib doesn't work well w/ PIL)
    composite = merge_stack(images)
    comparr = np.asarray(composite)

    # arrow calculations
    x, y = centroids[0]
    dx, dy = (c1 - c0 for c0, c1 in zip(*centroids))

    f, ax = plt.subplots()
    f.set_size_inches((10, 10))
    ax.set_aspect('equal')
    ax.set_title('Gap from id {} to {}, {:0.1f} px, {:0.3f} sec (EID: {})'.format(
            lost_bid, found_bid,
            math.sqrt(dx**2 + dy**2), times[1] - times[0],
            experiment.id))

    ax.fill([crop_box.left, crop_box.left, crop_box.right, crop_box.right],
            [crop_box.bottom, crop_box.top, crop_box.top, crop_box.bottom],
            hatch='////', facecolor='0.7', edgecolor='0.9',
            zorder=-10)

    ax.imshow(comparr, cmap=plt.cm.YlGn, extent=extents.vflip,
              interpolation='nearest')

    _, patches = patch_contours(shapes)
    for patch, color in zip(patches, ['red', 'blue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        ax.add_patch(patch)

    ar = 0.1, 0.9
    ax.arrow(x + ar[0]*dx, y + ar[0]*dy, (ar[1] - ar[0])*dx, (ar[1] - ar[0])*dy,
             width=1.5, head_length=6, head_width=4, length_includes_head=True,
             color='yellow', alpha=0.8)

    ax.set_xlim(crop_box.x)
    ax.set_ylim(crop_box.y)

    return f, ax
