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
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .. import tools
from . import pil
from .plotting import plot_spacetime, tweak_bounds

BOUNDS = {
    'padding': 40,
    'min_dim': 120,
}

def show_collision_choices(experiment, graph, target):
    parents = graph.predecessors(target)
    children = graph.successors(target)

    def term_data(bids, end):
        return tools.terminal_data(experiment, bids, itertools.repeat(end))
    pdata = term_data(parents, 'last')
    cdata = term_data(children, 'first')

    boxes = itertools.chain(pdata['bounds'], cdata['bounds'])
    bounds = tweak_bounds(sum(boxes), **BOUNDS)

    # should be the same times
    parent_times = set(pdata['time'])
    assert len(parent_times) == 1
    parent_time = parent_times.pop()

    children_times = set(cdata['time'])
    assert len(children_times) == 1
    children_time = children_times.pop()

    ppatches = tools.patch_contours(pdata['shape'], bounds=False)
    cpatches = tools.patch_contours(cdata['shape'], bounds=False)

    print(ppatches)

    f = plt.figure()
    f.set_size_inches((14, 10))

    gs = gridspec.GridSpec(nrows=2, ncols=6,
                           height_ratios=[2.5, 3],
                           width_ratios=None)
    ax_before = f.add_subplot(gs[0, 0:2])
    ax_during = f.add_subplot(gs[0, 2:4])
    ax_after = f.add_subplot(gs[0, 4:6])
    ax_options = [f.add_subplot(gs[1, 0:3]), f.add_subplot(gs[1, 3:6])]

    # show before and after
    axs = [ax_before, ax_after]
    times = [parent_time, children_time]
    for ax, time, patches in zip(axs, times, (ppatches, cpatches)):
        plot_spacetime(ax, experiment, bounds, time)

        for patch in patches:
            patch.set_facecolor('red')
            patch.set_alpha(0.6)
            ax.add_patch(patch)

    return f, axs
