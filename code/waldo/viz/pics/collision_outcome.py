# -*- coding: utf-8 -*-
"""
MWT collision visualization (for screening)
"""
from __future__ import absolute_import, division, print_function
import six
from six.moves import zip, range

import math
from copy import copy
import functools
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .. import tools
from . import pil
from . import plotting as wplt

BOUNDS = {
    'padding': 20,
    'min_dim': 120,
}

def show_collision_choices(experiment, graph, target, ident=False):
    """
    Make a figure showing the collision centered on *target*. Returns a
    figure & axis pair.

    If *ident* is true, return data about the pairings.
    """
    parents = sorted(graph.predecessors(target))
    children = sorted(graph.successors(target))

    def term_data(bids, end):
        return tools.terminal_data(experiment, bids, itertools.repeat(end))
    pdata = term_data(parents, 'last')
    cdata = term_data(children, 'first')

    boxes = itertools.chain(pdata['bounds'], cdata['bounds'])
    bounds = wplt.tweak_bounds(sum(boxes), **BOUNDS)

    # should be the same times
    def the_one(l):
        d = set(l)
        assert len(d) == 1
        return d.pop()
    parent_time = the_one(pdata['time'])
    parent_frame = the_one(pdata['frame'])
    children_time = the_one(cdata['time'])
    children_frame = the_one(cdata['frame'])

    ppatches = tools.patch_contours(pdata['shape'], bounds=False)
    cpatches = tools.patch_contours(cdata['shape'], bounds=False)

    f = plt.figure()
    height = 12
    aspect = 8/7
    f.set_size_inches((height * aspect, height))

    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 3], hspace=0.1)
    gsp = functools.partial(gridspec.GridSpecFromSubplotSpec, nrows=1)
    gs_top = gsp(ncols=3, subplot_spec=gs[0], wspace=0.02)
    gs_bot = gsp(ncols=2, subplot_spec=gs[1])

    ax_before = f.add_subplot(gs_top[0])
    ax_more = [f.add_subplot(gs_top[i], sharey=ax_before) for i in range(1, 3)]
    for ax in ax_more:
        plt.setp(ax.get_yticklabels(), visible=False)
    ax_during, ax_after = ax_more

    ax_options = [f.add_subplot(gs_bot[i]) for i in range(2)]

    # show before and after
    axs = [ax_before, ax_after]
    times = [parent_time, children_time]
    frames = [parent_frame, children_frame]
    cmaps = [plt.cm.winter, plt.cm.autumn]
    blobs = [parents, children]
    for ax, time, frame, patches, cmap, bids in zip(
            axs, times, frames, (ppatches, cpatches), cmaps, blobs):
        wplt.plot_spacetime(ax, experiment, bounds, time=time)

        for patch, cidx in zip(patches, np.linspace(0.2, 0.8, len(patches))):
            patch.set_facecolor(cmap(cidx))
            patch.set_alpha(0.6)
            ax.add_patch(patch)

        ax.set_title('Frame {}, Blobs: {}'.format(frame, tools.csl(bids)))

    # stack up the collision images
    stack_axs = itertools.chain((ax_during,), ax_options)
    wplt.plot_spacetime_rainbow(stack_axs, experiment, bounds,
            inverted=True,
            time=[parent_time, children_time],
        )
    ax_during.set_title('Frames {} to {}'.format(parent_frame, children_frame))

    # define pairs
    pair_groups = [[(0, 0), (1, 1)],
                   [(0, 1), (1, 0)]]

    if ident:
        ident_data = {}
        for side, pairs in zip(['left', 'right'], pair_groups):
            ident_data[side] = [(parents[x], children[y]) for x, y in pairs]

    # draw matching contours for each pair
    for ax, pairs in zip(ax_options, pair_groups):
        ppatches2 = tools.patch_contours(pdata['shape'], bounds=False)
        cpatches2 = tools.patch_contours(cdata['shape'], bounds=False)
        for (pp, cp), color, hatch in zip(pairs, ['lime', 'magenta'], ['\\'*2, r'/'*2]):
            patch_pair = [ppatches2[pp], cpatches2[cp]]
            for patch in patch_pair:
                patch.set_fill(False)
                patch.set_edgecolor(color)
                patch.set_alpha(0.85)
                patch.set_linewidth(2)
                patch.set_hatch(hatch)
                ax.add_patch(patch)

        # zoom in a little
        b = tools.Box(x=ax.get_xlim(), y=ax.get_ylim())
        b.grow(-BOUNDS['padding'] + 10) # reduce padding to something else
        ax.set_xlim(b.x)
        ax.set_ylim(b.y)

    # draw arrows for each pair
    #### WIP ####################################################

    if ident:
        return f, axs, ident_data
    else:
        return f, axs
