# -*- coding: utf-8 -*-
"""
Just show a single blob, where it starts and stops (for screening)
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
MIN_DIM = 120
ASPECT = 2/3

def show_blob(experiment, bid):
    bids = [bid, bid]
    ends = ['first', 'last']
    data = tools.terminal_data(experiment, bids, ends)

    tweaker = functools.partial(tweak_bounds,
                    padding=PADDING, min_dim=MIN_DIM, aspect=ASPECT)
    bounds = [tweaker(b) for b in data['bounds']]

    images, extents = zip(*(
        pil.load_image_portion(experiment, b, time=t)
        for t, b
        in zip(data['time'], bounds)))

    _, patches = tools.patch_contours(data['shape'])

    f, axs = plt.subplots(ncols=2)
    f.set_size_inches((14, 10))
    f.tight_layout()
    f.subplots_adjust(top=0.91, wspace=0)

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
            prefs, axs, bounds, images, extents, patches, data['time']):
        ax.set_title(p['title'].format(t), fontsize=14)
        show_image(ax, img, ext, box, cmap=plt.cm.Blues)

        patch.set_facecolor(p['color'])
        patch.set_alpha(0.6)
        ax.add_patch(patch)

    return f, axs
