# -*- coding: utf-8 -*-
"""
Just show a single blob, where it starts and stops (for screening)
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

def show_image(axes, image, extents, limits, cmap=plt.cm.YlGn):
    axes.set_aspect('equal')

    axes.fill([limits.left, limits.left, limits.right, limits.right],
              [limits.bottom, limits.top, limits.top, limits.bottom],
        hatch='////', facecolor='0.7', edgecolor='0.9',
        zorder=-10)

    axes.imshow(np.asarray(image), cmap=cmap, extent=extents.vflip,
              interpolation='nearest')

    axes.set_xlim(limits.x)
    axes.set_ylim(limits.y)

def show_blob(experiment, bid):
    data = [
        tools.terminal_data(experiment, bid, end) for end in ['first', 'last']
    ]
    bounds = [d['bounds'] for d in data]
    shapes = [d['shape'] for d in data]
    times = [d['time'] for d in data]
    centroids = [d['centroid'] for d in data]

    # tweak bounds to make uniformly square images
    def tweak_bounds(bounds):
        bounds.grow(40)
        min_sq_dim = 120
        if bounds.width < min_sq_dim:
            bounds.width = min_sq_dim
        if bounds.height < min_sq_dim:
            bounds.height = min_sq_dim
        bounds.square()
        bounds.height = 1.5 * bounds.width # tweak aspect
        bounds.round()
        return bounds

    bounds = [tweak_bounds(b) for b in bounds]

    # load images and crop
    image_fns = [experiment.image_files.nearest(time=t)[0] for t in times]
    images = []
    extents = []
    for fn, box in zip(image_fns, bounds):
        im, ext = pil.crop(pil.load(str(fn)), box)
        # convert to array (matplotlib doesn't work well w/ PIL)
        images.append(pil.adjust(im))
        extents.append(ext)

    _, patches = tools.patch_contours(shapes)

    f, axs = plt.subplots(ncols=2)
    f.set_size_inches((15, 10))
    f.tight_layout()
    f.subplots_adjust(top=0.91, wspace=-0.15)

    prefs = [
        {
            'title': 'Start (t = {:0.3f} s)',
            'color': 'green',
        },
        {
            'title': 'End (t = {:0.3f} s)',
            'color': 'red',
        }
    ]
    f.suptitle('Blob ID {} (EID: {})'.format(bid, experiment.id), fontsize=20)
    for p, ax, box, img, ext, patch, t in zip(
            prefs, axs, bounds, images, extents, patches, times):
        ax.set_title(p['title'].format(t), fontsize=14)
        show_image(ax, img, ext, box, cmap=plt.cm.Blues)

        patch.set_facecolor(p['color'])
        patch.set_alpha(0.6)
        ax.add_patch(patch)

    return f, axs
