# -*- coding: utf-8 -*-
"""
Absorb small neigbors, connecting their edges
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

from ..util import frame_filter, condense_nodes

__all__ = [
    'assimilate',
]

def assimilate(digraph, max_threshold):
    """
    For the directed graph *digraph*, absorb neigbors that are shorter than
    *max_threshold*, given in the number of frames.

    For example, simple splits would be absorbed if they are small enough.

    ..
             large1
              /  \                 large1 (components: {small1, small2})
             /    \                  |
        small1    small2    ===>     |
             \    /                  |
              \  /                 large2
             large2


    More complex topology can also be handled via recursive steps

    ..
        [MISSING EXAMPLE]


    The degree of absorbed nodes in the direction towards the assimilator
    must be one.  That is to prevent contractions like the following

    ..  large   large
            \   /
             \ /
            small       | t
             / \        |
            /   \       V
        large   large

    Who would get the small node?  Nobody hopefully.
    """
    # copy because we'll be modifying it and modifying while iterating is
    # fraught with peril
    nodes = digraph.nodes()

    while nodes:
        node = nodes.pop()