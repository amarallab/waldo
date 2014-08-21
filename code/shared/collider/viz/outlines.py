# -*- coding: utf-8 -*-
"""
MWT collision graph visualizations - Before and after outlines
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy import ndimage

import multiworm

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
            except ValueError, IndexError:
                pass
            i += 1

    for parent in parents:
        i = -1
        while True:
            try:
                outline = multiworm.readers.blob.decode_outline_line(experiment[parent], i)
                parent_outlines.append(outline)
                break
            except ValueError, IndexError:
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

def patch_contours(contours):
    x_min, x_max, y_min, y_max = 10000, 0, 10000, 0
    patches = []
    for contour in contours:
        contour = np.array(contour)
        x_min = min(x_min, contour.T[0].min())
        x_max = max(x_max, contour.T[0].max())
        y_min = min(y_min, contour.T[1].min())
        y_max = max(y_max, contour.T[1].max())
        patches.append(Polygon(contour, closed=True))

    bounds = x_min, x_max, y_min, y_max
    return bounds, patches

def pad_bounds(bounds, padding):
    return bounds[0] - padding, bounds[1] + padding, bounds[2] - padding, bounds[3] + padding

def get_image(experiment, time, bounds):
    actual_time, im_file = experiment.image_files.nearest(time=time)
    img = ndimage.imread(str(im_file))
    return img

def show_collision(experiment, graph, target, direction='backwards'):
    backwards = direction.startswith('back')

    if backwards:
        others = graph.predecessors(target)
    else:
        others = graph.successors(target)
        raise NotImplementedError()

    blob = experiment[target]
    others = [experiment[other] for other in others]

    blob.df.decode_contour()
    for other in others:
        other.df.decode_contour()

    if backwards:
        blob_contour = blob.df['contour'][blob.df['contour'].first_valid_index()]
        other_contours = [other.df['contour'][other.df['contour'].last_valid_index()] for other in others]

    contours = [blob_contour]
    contours.extend(other_contours)

    bounds, patches = patch_contours(contours)
    collection_pre = PatchCollection(patches[1:], cmap=plt.cm.jet, alpha=0.3)
    collection_post = PatchCollection(patches[:1], facecolor='green', alpha=0.3)
    collection_pre.set_array(np.linspace(0.1, 0.9, len(patches) - 1))

    f, axs = plt.subplots(ncols=2)
    f.set_size_inches((10, 6))

    bounds = pad_bounds(bounds, 20)
    image = get_image(experiment, int(blob.born), 0)
    vmin, vmax = np.percentile(image, [6, 94])

    collections = [collection_pre, collection_post]
    for ax, coll in zip(axs, collections):
        ax.imshow(image.T, cmap=plt.cm.Greys, interpolation='none', vmin=vmin, vmax=vmax)
        ax.add_collection(coll)
        ax.set_xlim(*bounds[:2])
        ax.set_ylim(*bounds[2:])
        ax.set_aspect('equal')

    return f, axs
