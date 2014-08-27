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
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, Wedge
from matplotlib.collections import PatchCollection
from scipy import ndimage

import multiworm

from .box import Box

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
    bounds = []
    patches = []
    for contour in contours:
        contour = np.array(contour)
        if contour.shape == (2,):
            contour = np.reshape(contour, (-1, 2))
        bbox = Box.fit(contour)
        if len(contour) == 1:
            patches.append(Wedge(contour, 10, 0, 360, 5, alpha=0.6))
            bbox.size = 30, 30
        else:
            patches.append(Polygon(contour, closed=True, alpha=0.3))
        bounds.append(bbox)

    bounds = sum(bounds)
    return bounds, patches

def get_image(experiment, time, bounds):
    actual_time, im_file = experiment.image_files.nearest(time=time)
    img = ndimage.imread(str(im_file))
    return img

def get_contour(blob, index='first', centroid_fallback=True):
    if index not in ['first', 'last']:
        raise ValueError('index must be either "first" or "last"')

    method = index + '_valid_index'

    col = 'contour'
    index = getattr(blob.df[col], method)()
    if index is None:
        if not centroid_fallback:
            raise KeyError('No valid index')
        col = 'centroid'
        index = getattr(blob.df[col], method)()

    return blob.df[col][index]

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
        blob_contour = get_contour(blob, index='first')
        other_contours = [get_contour(other, index='last') for other in others if not other.empty]

    contours = [blob_contour]
    contours.extend(other_contours)

    bounds, patches = patch_contours(contours)
    collection_pre = PatchCollection(patches[1:], cmap=plt.cm.autumn, alpha=0.3)
    collection_post = PatchCollection(patches[:1], facecolor='green', alpha=0.3)
    collection_pre.set_array(np.linspace(0.3, 0.9, len(patches) - 1))

    f, axs = plt.subplots(ncols=2, sharey=True)
    f.set_size_inches((10, 6))
    #f.subplots_adjust(wspace=None)
    f.tight_layout()

    bounds.grow(20)
    bounds.width = max(120, bounds.width)
    bounds.height = max(180, bounds.height)
    #bounds.shape = 120, 180 # this seems to work well enough?

    image = get_image(experiment, blob.born_t, 0)
    vmin, vmax = np.percentile(image, [3, 97])

    collections = [collection_pre, collection_post]
    for ax, coll, blobs, frame in zip(axs, collections, [others, [blob]], [others[0].died_f, blob.born_f]):
        ax.imshow(image.T, cmap=plt.cm.Greys, interpolation='none', vmin=vmin, vmax=vmax)
        ax.add_collection(coll)
        ax.set_xlim(bounds.x)
        ax.set_ylim(bounds.y)
        ax.set_aspect('equal')
        ax.set_title('Blob(s): {}, Frame {}'.format(
                ', '.join(str(b.id) for b in blobs),
                int(frame)))

    return f, axs
