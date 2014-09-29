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

from .. import tools
from . import pil

def show_gap(experiment, lost_bid, found_bid):
    data = [
            tools.terminal_data(experiment, bid, end)
            for bid, end
            in zip([lost_bid, found_bid], ['last', 'first'])
        ]
    boxes = [d['bounds'] for d in data]
    shapes = [d['shape'] for d in data]
    times = [d['time'] for d in data]
    centroids = [d['centroid'] for d in data]

    # tweak bounds to make uniformly square images
    bounds = sum(boxes)
    bounds.grow(40)
    min_sq_dim = 150
    if bounds.width < min_sq_dim:
        bounds.width = min_sq_dim
    if bounds.height < min_sq_dim:
        bounds.height = min_sq_dim
    bounds.square()
    bounds.round()

    # load all images and crop
    image_fns = experiment.image_files.spanning(times=times)
    images = []
    for f in image_fns:
        im, extents = pil.crop(pil.load(str(f)), bounds)
        images.append(pil.adjust(im))

    # merge and convert to array (matplotlib doesn't work well w/ PIL)
    composite = pil.merge_stack(images)
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

    ax.fill([bounds.left, bounds.left, bounds.right, bounds.right],
            [bounds.bottom, bounds.top, bounds.top, bounds.bottom],
            hatch='////', facecolor='0.7', edgecolor='0.9',
            zorder=-10)

    ax.imshow(comparr, cmap=plt.cm.YlGn, extent=extents.vflip,
              interpolation='nearest')

    _, patches = tools.patch_contours(shapes)
    for patch, color in zip(patches, ['red', 'blue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        ax.add_patch(patch)

    ar = 0.1, 0.9
    ax.arrow(x + ar[0]*dx, y + ar[0]*dy, (ar[1] - ar[0])*dx, (ar[1] - ar[0])*dy,
             width=1.5, head_length=6, head_width=4, length_includes_head=True,
             color='yellow', alpha=0.8)

    ax.set_xlim(bounds.x)
    ax.set_ylim(bounds.y)

    return f, ax
