# -*- coding: utf-8 -*-
"""
Absorb small neigbors, connecting their edges
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import logging
L = logging.getLogger(__name__)

import networkx as nx

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

    ..
        large   large
            \   /
             \ /
            small       | t
             / \        |
            /   \       V
        large   large


    Who would get the small node?  Nobody hopefully.
    """
    #digraph.validate()
    methods = {
        'down': {
            'relatives': digraph.successors,
            'away_degree': digraph.out_degree,
            'towards_degree': digraph.in_degree,
        },
        'up': {
            'relatives': digraph.predecessors,
            'away_degree': digraph.in_degree,
            'towards_degree': digraph.out_degree,
        },
    }

    for dr, meth in six.iteritems(methods):
        # copy because we'll be modifying it and modifying while
        # iterating is fraught with peril
        nodes = digraph.nodes()
        nodes.sort()

        while nodes:
            node = nodes.pop()
            if node not in digraph:
                continue

            L.debug('basis node: {}'.format(node))
            relatives = set(meth['relatives'](node))
            while relatives:
                rnode = relatives.pop()
                L.debug('- checking relative {}'.format(rnode))

                # check exclusions
                if digraph.lifespan(rnode) > max_threshold:
                    L.debug(' - lifespan too short'.format(rnode))
                    continue
                if meth['towards_degree'](rnode) != 1:
                    L.debug(' - towards degree not 1'.format(rnode))
                    continue

                # new relatives after condensing
                relatives.update(meth['relatives'](rnode))

                # assimilate relative
                L.debug(' - absorbing node {} into {}'.format(rnode, node))
                digraph.condense_nodes(node, rnode)


