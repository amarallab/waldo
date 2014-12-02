# -*- coding: utf-8 -*-
"""
MWT collision visualization (for screening)
"""
from __future__ import absolute_import, division, print_function
import six
from six.moves import zip, range

import math

import numpy as np
import matplotlib.pyplot as plt

from .. import tools
from . import pil
from .plotting import plot_spacetime, tweak_bounds

BOUNDS = {
    'padding': 40,
    'min_dim': 150,
}

def show_gap(experiment, lost_bid, found_bid):
    bids = [lost_bid, found_bid]
    ends = ['last', 'first']
    data = tools.terminal_data(experiment, bids, ends)

    # determine space
    space = tweak_bounds(sum(data['bounds']), **BOUNDS)
    time = data['time']

    # arrow calculations
    x, y = data['centroid'][0]
    dx, dy = (c1 - c0 for c0, c1 in zip(*data['centroid']))

    f, ax = plt.subplots()
    f.set_size_inches((10, 10))
    ax.set_title('Gap from id {} to {}, {:0.1f} px, {:0.3f} sec (EID: {})'.format(
            lost_bid, found_bid,
            math.sqrt(dx**2 + dy**2), time[1] - time[0],
            experiment.id))

    plot_spacetime(ax, experiment, space, time=time, cmap=plt.cm.YlGn)

    patches = tools.patch_contours(data['shape'], bounds=False)
    for patch, color in zip(patches, ['red', 'blue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        ax.add_patch(patch)

    ar = 0.1, 0.9
    ax.arrow(x + ar[0]*dx, y + ar[0]*dy, (ar[1] - ar[0])*dx, (ar[1] - ar[0])*dy,
             width=1.5, head_length=6, head_width=4, length_includes_head=True,
             color='yellow', alpha=0.8)

    return f, ax
