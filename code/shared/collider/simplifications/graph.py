# -*- coding: utf-8 -*-
"""
Manipulations removing degree-one things (short offshoots and basic chains)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import networkx as nx

class ColliderGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super(ColliderGraph, self).__init__(*args, **kwargs)

        self._whereis_data = {}

    def copy(self):
        return ColliderGraph(self)

    def components(self, node):
        return set(int(n) for n in self.node[node].get('components', [node]))

    def where_is(self, bid):
        return self._whereis_data.get(bid, bid)

    def condense_nodes(self, node, *other_nodes):
        """
        Incorporate all nodes in *other_nodes* into *node* in-place.

        All nodes must have ``born_f`` and ``died_f`` keys, the condensed node
        born/died keys will maximize age.  All other key values will be used to
        ``.update()`` the condensed node data, so sets and mappings
        (dictionary-like objects) can be used.  The ``components`` key is
        slightly special, it will be created as a 1-element set with the
        node's name if non-existent.

        Edges that *other_nodes* had will be taken by *node*, excepting those to
        *node* to prevent self-loops.
        """
        node_data = self.node[node]

        # keep track of lifespans to check if we are removing long tracks
        orig_lifespan = round(self.lifespan_t(node), ndigits=2)
        lifespan_log = {}
        born_log, died_log = {}, {}

        born_log[node]= round(node_data['born_t'], ndigits=2)
        died_log[node]= round(node_data['died_t'], ndigits=2)

        #print('condensing', node, node_data)
        if 'components' not in node_data:
            node_data['components'] = set([node])


        for other_node in other_nodes:
            other_data = self.node[other_node]

            lifespan_log[other_node]= self.lifespan_t(other_node)
            born_log[other_node]= other_data['born_t']
            died_log[other_node]= other_data['died_t']

            # abscond with born/died
            node_data['born_f'] = min(node_data['born_f'], other_data.pop('born_f'))
            node_data['died_f'] = max(node_data['died_f'], other_data.pop('died_f'))

            node_data['born_t'] = min(node_data['born_t'], other_data.pop('born_t'))
            node_data['died_t'] = max(node_data['died_t'], other_data.pop('died_t'))

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

        final_lifespan = self.lifespan_t(node)

        longer_lifespans = [] #l for l in lifespan_log.values() if l > 60 * 20]
        no_lifespan_extentions = [] #l for l in lifespan_log.values() if l > final_lifespan]
        for n, life in lifespan_log.iteritems():
            if life > 60 * 20 and life + orig_lifespan > final_lifespan:
                longer_lifespans.append(n)
            if life > final_lifespan:
                no_lifespan_extentions.append(n)

        if len(longer_lifespans) or len(no_lifespan_extentions):
            if longer_lifespans:
                print('blob {id} merging with {n} other nodes'.format(id=int(node), n=len(other_nodes)))
                l1, l2 = orig_lifespan, final_lifespan
                print('\tWARNING: {n} blobs with longer lifespans removed'.format(n=len(longer_lifespans)))
                print('\t{l1}s --> {l2}s. (gain={g}s)'.format(l1=l1, l2=l2, g=l2-l1))

                #print('\tid, born, died, lifespan')
                #print('\t', node, born_log[node], died_log[node], orig_lifespan)
                'lifespan groupings'
                for n in longer_lifespans:
                    print('\t {id} | {l} + {l1} = {l2}'.format(id=int(n), l=lifespan_log[n],
                                                               l1=l1, l2=lifespan_log[n] + l1))

                print('\t merged node: born {b}-- died {d}'.format(b=node_data['born_t'], d=node_data['died_t']))
                b1 = born_log[node]
                d1 = died_log[node]
                for n in longer_lifespans:
                    b, d = born_log[n], died_log[n]
                    if b < b1:
                        overlap = (d - b1)
                        if overlap <0:
                            overlap = 0
                        print('\t({b}-{d}) and ({b1}-{d1}) | overlap={o}s'.format(b=b, d=d, b1=b1, d1=d1, o=overlap))

                    else:
                        overlap = (d1 - b)
                        if overlap <0:
                            overlap = 0
                        print('\t({b1}-{d1}) and ({b}-{d}) | overlap={o}s'.format(b=b, d=d, b1=b1, d1=d1, o=overlap))

            if no_lifespan_extentions:
                print('WARNING: {n} blobs with lifespans > 20 min'.format(n=len(no_lifespan_extentions)))
                print(lifespan_log)

        # update what's where
        for component in node_data['components']:
            self._whereis_data[component] = node


    def validate(self):
        """
        Verify that the graph contains the requisite data
        """
        for node, node_data in self.nodes_iter(data=True):
            for req_key in ['born_f', 'died_f', 'born_t', 'died_t']:
                if req_key not in node_data:
                    raise AssertionError("Node {} missing required key '{}'".format(node, req_key))

    def lifespan_f(self, node):
        # +1 because something that was born & died on the same frame exists
        # for 1 frame.
        return self.node[node]['died_f'] - self.node[node]['born_f'] + 1

    def lifespan_t(self, node):
        # This is off by one frame compared to the frame-based lifespan
        # because we don't know for sure how long the frame lasted
        return self.node[node]['died_t'] - self.node[node]['born_t']
