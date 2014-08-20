# -*- coding: utf-8 -*-
"""
Manipulations removing degree-one things (short offshoots and basic chains)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import networkx as nx

# class ColliderGraph(object):
#     def __init__(self, digraph):
#         self._digraph = digraph

#     def __getattribute__(self, name):
#         cls_attrs = ['_digraph', 'whereis', 'copy']
#         if name not in cls_attrs:
#             return object.__getattribute__(object.__getattribute__(self, '_digraph'), name)
#         else:
#             return object.__getattribute__(self, name)
#         #return getattr(self._digraph, name)

#     def whereis(self, node):
#         """find a node inside other nodes of the digraph"""

#     def copy(self):
#         return ColliderGraph(self._digraph.copy())

class ColliderGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super(ColliderGraph, self).__init__(*args, **kwargs)

        self._whereis_data = {}

    def copy(self):
        return ColliderGraph(self)

    def where_is(self, bid):
        """
        What
        """
        return self._whereis_data.get(bid, bid)

    def condense_nodes(self, node, *other_nodes):
        """
        Incorporate all nodes in *other_nodes* into *node* in-place.

        All nodes must have ``born`` and ``died`` keys, the condensed node
        born/died keys will maximize age.  All other key values will be used to
        ``.update()`` the condensed node data, so sets and mappings
        (dictionary-like objects) can be used.  The ``components`` key is
        slightly special, it will be created as a 1-element set with the
        node's name if non-existent.

        Edges that *other_nodes* had will be taken by *node*, excepting those to
        *node* to prevent self-loops.
        """

        node_data = self.node[node]
        if 'components' not in node_data:
            node_data['components'] = set([node])

        for other_node in other_nodes:
            self._whereis_data[other_node] = node
            other_data = self.node[other_node]

            # abscond with born/died
            node_data['born'] = min(node_data['born'], other_data.pop('born'))
            node_data['died'] = max(node_data['died'], other_data.pop('died'))

            # combine set/mapping data
            node_data['components'].update(
                    other_data.pop('components', set([other_node])))
            for k, v in six.iteritems(other_data):
                if k in node_data:
                    node_data[k].update(v)
                else:
                    node_data[k] = v

            # transfer edges
            self.add_edges_from(
                    (node, out_node)
                    for out_node
                    in self.successors(other_node)
                    if out_node != node)
            self.add_edges_from(
                    (in_node, node)
                    for in_node
                    in self.predecessors(other_node)
                    if in_node != node)

            # remove node
            self.remove_node(other_node)

    def validate(self):
        """
        Verify that the graph contains the requisite data
        """
        for node, node_data in self.nodes_iter(data=True):
            for req_key in ['born', 'died']:
                if req_key not in node_data:
                    raise AssertionError("Node {} missing required key '{}'".format(node, req_key))

    def lifespan(self, node):
        return self.node[node]['died'] - self.node[node]['born']
