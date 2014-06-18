# -*- coding: utf-8 -*-
"""
MWT collision graph visualizations - Before and after outlines
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools

import matplotlib.pyplot as plt

import multiworm

def before_and_after(experiment, target):
    """
    Find all children of target, then find all parents of the children.  Draw the last available outline
    just before the parents become the children, and the first available outline of the children.
    """
    digraph = experiment.collision_graph
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
            except ValueError:
                pass
            i += 1

    for parent in parents:
        i = -1
        while True:
            try:
                outline = multiworm.readers.blob.decode_outline_line(experiment[parent], i)
                parent_outlines.append(outline)
                break
            except ValueError:
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
