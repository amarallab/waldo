# -*- coding: utf-8 -*-
"""
Just show a single blob, where it starts and stops (for screening)
"""
from __future__ import absolute_import, division, print_function
import six
from six.moves import zip, range

import math
import functools

import numpy as np
import matplotlib.pyplot as plt

from .. import tools
from . import pil
from .plotting import plot_spacetime, tweak_bounds

BOUNDS = {
    'padding': 40,
    'min_dim': 120,
    'aspect': 3/4,
}

def show_blob(experiment, bid):
    bids = [bid, bid]
    ends = ['first', 'last']
    data = tools.terminal_data(experiment, bids, ends)

    tweaker = functools.partial(tweak_bounds, **BOUNDS)
    spaces = [tweaker(b) for b in data['bounds']]
    times = data['time']

    patches = tools.patch_contours(data['shape'], bounds=False)

    f, axs = plt.subplots(ncols=2)
    f.set_size_inches((14, 10))
    f.tight_layout()
    f.subplots_adjust(top=0.94)

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
    for p, ax, space, time, patch in zip(prefs, axs, spaces, times, patches):
        ax.set_title(p['title'].format(time), fontsize=14)
        plot_spacetime(ax, experiment, space, time=time, cmap=plt.cm.Blues)

        patch.set_facecolor(p['color'])
        patch.set_alpha(0.6)
        ax.add_patch(patch)

    return f, axs
