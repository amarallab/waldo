# -*- coding: utf-8 -*-
"""
MWT collision graph visualizations - Before and after outlines
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

import multiworm
from . import tools as vt

def before_and_after(experiment, target):
    """
    Find all children of target, then find all parents of the children.  Draw the last available outline
    just before the parents become the children, and the first available outline of the children.
    """
    digraph = experiment.graph
    children = list(digraph.successors(target))
    parents = list(itertools.chain.from_iterable(digraph.predecessors_iter(c) for c in children))
    #return parents, children

    children_outlines = []
    parent_outlines = []
    for child in children:
        i = 0
        while True:
            try:
                outline = multiworm.readers.blob.decode_outline_line(experiment[child], i)
                children_outlines.append(outline)
                break
            except (ValueError, IndexError):
                pass
            i += 1

    for parent in parents:
        i = -1
        while True:
            try:
                outline = multiworm.readers.blob.decode_outline_line(experiment[parent], i)
                parent_outlines.append(outline)
                break
            except (ValueError, IndexError):
                pass
            i += -1
    return parent_outlines, children_outlines

def show_before_and_after(experiment, target):
    parents, children = before_and_after(experiment, target)

    f, axs = plt.subplots(ncols=2)
    for ax in axs:
        ax.axis('equal')
    for p in parents:
        axs[0].plot(*zip(*p), lw=3)
    for c in children:
        axs[1].plot(*zip(*c), lw=3)

    return f, axs

def show_around(experiment, frame, bounds):
    # get all blob outlines on given frame
    df = vt.frame_dataframe(experiment, frame)
    df = vt.fill_empty_contours(df)
    df = df.reindex(np.random.permutation(df.index))
    _, patches = vt.patch_contours(df['contour'])
    pc = PatchCollection(patches, cmap=plt.cm.winter, alpha=0.3)
    pc.set_array(np.linspace(0, 1, len(patches) - 1))

    # load image
    image = vt.get_image(experiment, frame=frame)
    contrast_boost = 3
    vmin, vmax = np.percentile(image, [contrast_boost, 100 - contrast_boost])

    # tweak bounds
    if isinstance(bounds, vt.Box):
        pass
    elif isinstance(bounds, collections.Mapping):
        bounds = vt.Box(**bounds)
    elif isinstance(bounds, collections.Iterable):
        bounds = vt.Box(*bounds)
    else:
        bounds = vt.Box(bounds)

    # plot
    f, ax = plt.subplots()
    f.set_size_inches((10, 10))
    ax.imshow(image.T, cmap=plt.cm.Greys, interpolation='none', vmin=vmin, vmax=vmax)
    ax.add_collection(pc)

    # annotate blobs
    for ix, row in df.iterrows():
        centroid = row['centroid']
        if centroid in bounds:
            ax.text(*centroid, s=str(row['bid']),
                    family='monospace', color='red', size='20',
                    weight='bold', verticalalignment='center',
                    horizontalalignment='center')

    ax.set_title('Experiment {}, Frame {}'.format(experiment.id, int(frame)))
    ax.set_xlim(bounds.x)
    ax.set_ylim(bounds.y)
    return f, ax, bounds

def show_terminal(experiment, target, end='tail', bounds=None):
    "Show what's near the given target from the experiment"
    # find target time and place
    target = experiment[target]
    if target.empty:
        raise ValueError('target is an empty blob')

    if end == 'tail':
        frame = target.died_f
        place = target['centroid'][-1]
    else:
        frame = target.born_f
        place = target['centroid'][0]

    bounds = bounds if bounds else {'center': place, 'size': (200, 200)}
    return show_around(experiment, frame, bounds=bounds) + (frame,)
