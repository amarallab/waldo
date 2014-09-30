# -*- coding: utf-8 -*-
"""
MWT collision visualization (for screening)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import zip, range

import math
import functools

import numpy as np
import matplotlib.pyplot as plt

from .. import tools
from . import pil
from .plotting import show_image, tweak_bounds

PADDING = 40
MIN_DIM = 150
ASPECT = 1

def show_gap(experiment, lost_bid, found_bid):
    bids = [lost_bid, found_bid]
    ends = ['last', 'first']
    data = tools.terminal_data(experiment, bids, ends)

    tweaker = functools.partial(tweak_bounds,
                    padding=PADDING, min_dim=MIN_DIM, aspect=ASPECT)

    bounds = tweaker(sum(data['bounds']))

    # load all images and crop
    images, extents = zip(*(
        pil.load_image_portion(experiment, bounds, filename=fn)
        for fn
        in experiment.image_files.spanning(times=data['time'])))

    # merge stack
    composite = pil.merge_stack(images)
    extents = extents[0] # should all be identical

    # arrow calculations
    x, y = data['centroid'][0]
    dx, dy = (c1 - c0 for c0, c1 in zip(*data['centroid']))

    f, ax = plt.subplots()
    f.set_size_inches((10, 10))
    ax.set_aspect('equal')
    ax.set_title('Gap from id {} to {}, {:0.1f} px, {:0.3f} sec (EID: {})'.format(
            lost_bid, found_bid,
            math.sqrt(dx**2 + dy**2), data['time'][1] - data['time'][0],
            experiment.id))

    show_image(ax, composite, extents, bounds)

    _, patches = tools.patch_contours(data['shape'])
    for patch, color in zip(patches, ['red', 'blue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        ax.add_patch(patch)

    ar = 0.1, 0.9
    ax.arrow(x + ar[0]*dx, y + ar[0]*dy, (ar[1] - ar[0])*dx, (ar[1] - ar[0])*dy,
             width=1.5, head_length=6, head_width=4, length_includes_head=True,
             color='yellow', alpha=0.8)

    return f, ax
