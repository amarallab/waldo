# -*- coding: utf-8 -*-
"""
Manipulations removing degree-one things (short offshoots and basic chains)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import logging

import networkx as nx

from conf import settings
from .network_number_wizard import network_number_wizard
from .network_number_wizard import node_is_moving

if settings.DEBUG:
    import inspect
    import pickle
    import random
    import string

L = logging.getLogger(__name__)

class ColliderGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        self.experiment = None

        ## including this causes major problems with the to_undirected()
        ## function ... not worth it.
        #self.experiment = kwargs.get('experiment', None)

        super(ColliderGraph, self).__init__(*args, **kwargs)
        self._whereis_data = {}


    def copy(self):
        return ColliderGraph(self, experiment=self.experiment)

    def components(self, node):
        return set(int(n) for n in self.node[node].get('components', [node]))

    def where_is(self, bid):
        return self._whereis_data.get(bid, bid)

    def _debug_dump(self):
        if not settings.DEBUG:
            raise RuntimeError("cannot run without DEBUG being set")

        word = ''.join(random.choice(string.lowercase) for _ in range(10))
        fname = 'graph_{}.pkl'.format(word)
        pickle.dump(self, fname)
        L.error('Dumped graph to {}'.format(fname))

    def condense_nodes(self, node, *other_nodes, **kwargs):
        """
        Incorporate all nodes in *other_nodes* into *node* in-place.

        All nodes must have ``born_f`` and ``died_f`` keys, the condensed node
        born/died keys will maximize age.  All other key values will be used to
        ``.update()`` the condensed node data, so sets and mappings
        (dictionary-like objects) can be used.  The ``components`` key is
        slightly special, it will be created as a 1-element set with the
        node's name if non-existent.

        Edges that *other_nodes* had will be taken by *node*, except those to
        *node* to prevent self-loops.

        If *skip_life_recalc* is True, born/died times on *node* will not be
        changed. Use with care.
        """
        skip_life_recalc = kwargs.get('skip_life_recalc', False)
        nd = self.node[node]
        L.debug('Node {} incorporating {}'.format(
                node, ', '.join(str(x) for x in other_nodes)))


        subg = self.subgraph((node,) + other_nodes)
        if nx.number_connected_components(subg.to_undirected()) != 1:
            raise ValueError('Attempting to merge unconnected nodes.')

        # for a, b in subg.edges_iter():
        #     if subg.node[a]['died_f'] != subg.node[b]['born_f'] - 1:
        #         raise ValueError('Non-concurrnet')


        # was an important precaution... bug probably fixed.
        #if node in other_nodes:
        #    other_nodes = list(other_nodes)
        #    other_nodes.pop(other_nodes.index(node))

        #####
        if settings.DEBUG:
            # warning: kinda slow
            #curframe = inspect.currentframe()
            #calframe = inspect.getouterframes(curframe, 2)
            #L.debug('Requestor function: {}'.format(calframe[1][3]))

            lifespans = {n: self.lifespan_f(n) for n in other_nodes}
            orig_lifespan = self.lifespan_f(node)

            births = {n: self.node[n]['born_f'] for n in other_nodes}
            deaths = {n: self.node[n]['died_f'] for n in other_nodes}

            births[node] = nd['born_f']
            deaths[node] = nd['died_f']
        #####

        if 'components' not in nd:
            nd['components'] = set([node])

        for other_node in other_nodes:
            other_data = self.node[other_node]

            # abscond with born/died
            if not skip_life_recalc:
                nd['born_f'] = min(nd['born_f'], other_data.pop('born_f'))
                nd['died_f'] = max(nd['died_f'], other_data.pop('died_f'))

                nd['born_t'] = min(nd['born_t'], other_data.pop('born_t'))
                nd['died_t'] = max(nd['died_t'], other_data.pop('died_t'))
            else:
                for key in ('born_f', 'died_f', 'born_t', 'died_t'):
                    del other_data[key]

            # combine set/mapping data
            nd['components'].update(
                    other_data.pop('components', set([other_node])))
            for k, v in six.iteritems(other_data):
                if k in nd:
                    nd[k].update(v)
                else:
                    nd[k] = v

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

        # update what's where
        for component in nd['components']:
            self._whereis_data[component] = node

        #####
        if settings.DEBUG:
            final_lifespan = self.lifespan_f(node)

            longer_lifespans = []
            no_lifespan_extentions = []
            for n, life in lifespans.iteritems():
                if life > 60 * 20 * 15 and life + orig_lifespan > final_lifespan:
                    longer_lifespans.append(n)
                if life > final_lifespan:
                    no_lifespan_extentions.append(n)

            if longer_lifespans:
                L.warn('Requestor function: {}'.format(calframe[1][3]))
                L.warn('Blob {} merging with {} other nodes'.format(
                        node, len(other_nodes)))
                for n in (node,) + other_nodes:
                    L.warn('Node {:6}, born_f {:6}, died_f {:6}'.format(
                            int(n), births[n], deaths[n]))
                l1, l2 = orig_lifespan, final_lifespan
                L.warn('\tWARNING: {} blobs with longer lifespans removed'.format(
                        len(longer_lifespans)))
                L.warn('\t{} fr --> {} fr. (gain={} fr)'.format(l1, l2, l2 - l1))

                for n in longer_lifespans:
                    L.warn('\t {} | {} + {} = {}'.format(
                        int(n), lifespans[n], l1, lifespans[n] + l1))

                L.warn('\t merged node: born {} -- died {}'.format(
                        nd['born_f'], nd['died_f']))
                b1 = births[node]
                d1 = deaths[node]
                for n in longer_lifespans:
                    b, d = births[n], deaths[n]
                    if b < b1:
                        overlap = (d - b1)
                        if overlap <0:
                            overlap = 0
                        L.warn('\t({}-{}) and ({}-{}) | overlap={} fr'.format(
                                b, d, b1, d1, overlap))

                    else:
                        overlap = (d1 - b)
                        if overlap <0:
                            overlap = 0
                        L.warn('\t({}-{}) and ({}-{}) | overlap={} fr'.format(
                                b, d, b1, d1, overlap))

            elif no_lifespan_extentions:
                L.warn('WARNING: {} blobs with lifespans > 20 min'.format(
                        len(no_lifespan_extentions)))
                L.warn(lifespans)
        #####

    def validate(self, acceptable_f_delta=-10):
        """
        Verify that the graph:

        * contains the requisite data attached to each node
        * contains only causal links
        """
        for node, node_data in self.nodes_iter(data=True):
            for req_key in ['born_f', 'died_f', 'born_t', 'died_t']:
                if req_key not in node_data:
                    raise AssertionError("Node {} missing required key '{}'".format(
                            node, req_key))

        for a, b in self.edges_iter():
            f_delta = self.node[b]['born_f'] - self.node[a]['died_f']
            if f_delta < acceptable_f_delta:
                raise AssertionError("Edge from {} to {} is acausal, going "
                        "back in time {:0.0f} frames".format(a, b, -f_delta))

        L.warn('Validation pass')

    def lifespan_f(self, node):
        # +1 because something that was born & died on the same frame exists
        # for 1 frame.
        return self.node[node]['died_f'] - self.node[node]['born_f'] + 1

    def lifespan_t(self, node):
        # This is off by one frame compared to the frame-based lifespan
        # because we don't know for sure how long the frame lasted
        return self.node[node]['died_t'] - self.node[node]['born_t']

    def count_worms(self, experiment):
        worm_count_dict = network_number_wizard(self, experiment)
        for node, count in worm_count_dict.iteritems():
            self.node[node]['worm_count'] = count

    def determine_moving(self, experiment):
        # code following network_number_wizard logic for determining
        # seeds
        terminals_df = experiment.prepdata.load('terminals')
        for node in self.nodes():
            is_moving = 0
            if node_is_moving(node, terminals_df):
                is_moving = 1
            self.node[node]['moved'] = is_moving

    def add_node_attributes(self, attribute_name, node_dict, default=False):
        """
        adds a new attribute to a set of nodes in the graph and passes
        values to set the attribute to.

        params
        -----
        attribute_name: (str)
            the key used to store the new attribute in node_data
        node_dict: (dict)
            a dictionary with {node: value} pairs specifying what
            the new attribute value is for each node.
        default:
            the default value for nodes that are not in the node_dict.
            if default == False, no attribute will be created.
        """
        nodes_found = 0
        nodes_not_found = 0
        for node in self.nodes():
            #print(type(node))
            #print(node, node in node_dict)
            if node in node_dict:
                self.node[node][attribute_name] = node_dict[node]
                nodes_found += 1
            elif default == False: # do not add anything
                nodes_not_found += 1
            else: # add a None placeholder
                nodes_not_found += 1
                self.node[node][attribute_name] = default
        print(nodes_found, 'nodes found')
        print(nodes_not_found, 'nodes not found')
