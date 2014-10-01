# -*- coding: utf-8 -*-
"""
Show a union of two blobs
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import zip, range

import math
import itertools
import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from .. import tools
from . import pil
from .plotting import plot_spacetime, tweak_bounds

BOUNDS = {
    'padding': 20,
    'min_dim': 120,
    'aspect': 3/4,
}

def show_collision(experiment, graph, target, direction='backwards'):
    backwards = direction.startswith('back')
    if backwards:
        target_end = 'first'
        relatives_end = 'last'
        relatives = graph.predecessors(target)
    else:
        target_end = 'last'
        relatives_end = 'first'
        relatives = graph.successors(target)

    def term_data(bids, end):
        return tools.terminal_data(experiment, bids, itertools.repeat(end))
    target_data = term_data([target], target_end)
    relative_data = term_data(relatives, relatives_end)

    boxes = itertools.chain(target_data['bounds'], relative_data['bounds'])
    space = tweak_bounds(sum(boxes), **BOUNDS)

    times = [target_data['time'][0], relative_data['time'][0]]
    frames = [target_data['frame'][0], relative_data['frame'][0]]

    def make_collection(data, **kwargs):
        patches = tools.patch_contours(data['shape'], bounds=False)
        return PatchCollection(patches, **kwargs)

    collections = [
        make_collection(target_data, facecolor='green', alpha=0.3),
        make_collection(relative_data, cmap=plt.cm.autumn, alpha=0.3),
    ]
    collections[1].set_array(np.linspace(0.3, 0.9, len(relatives)))

    bid_groups = [[target], relatives]

    f, axs = plt.subplots(ncols=2)
    f.set_size_inches((14, 10))
    f.tight_layout()
    f.subplots_adjust(top=0.94)
    f.suptitle('Collision Blob ID {} (EID: {})'.format(
            target, experiment.id), fontsize=20)

    if backwards:
        axs = reversed(axs)

    for ax, t, fr, coll, bids in zip(axs, times, frames, collections, bid_groups):
        plot_spacetime(ax, experiment, space, t)
        ax.add_collection(coll)
        ax.set_title('Blob(s): {}, Frame {}'.format(
                ', '.join(str(b) for b in bids), fr),
            fontsize=14)

    return f, axs
