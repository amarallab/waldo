# -*- coding: utf-8 -*-
"""
Manipulations removing degree-one things (short offshoots and basic chains)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import logging

# third party
import networkx as nx
import pandas as pd

# package specific
from waldo.conf import settings

from . import keyconsts as kc

from .network_number_wizard import node_is_moving, network_number_wizard

if settings.DEBUG:
    import inspect
    import pickle
    import random
    import string

L = logging.getLogger(__name__)

def merge_mappings(a, b):
    """
    MutableMapping *b* is merged into MutableMapping *a*, modifying *a*
    in-place.  Values in *a* and *b* are expected to be combinable...i.e.
    both lists, or sets, or dicts
    """
    for k, v in six.iteritems(b):
        if k in a:
            # try .update(); works for MutableMappings (e.g. dict) and
            # MutableSets (set).
            try:
                a[k].update(v)
            except TypeError:
                raise TypeError('Cannot update key "{}" with type {}, incompatible type provided: {}'
                        .format(k, type(a[k]), type(v)))
            except AttributeError:
                # maybe it's a list? try below...
                pass
            else:
                # OK, move on.
                continue

            # try .extend(); works for MutableSequences (list)
            try:
                a[k].extend(v)
            except TypeError:
                raise TypeError('Cannot extend key "{}" with type {}, incompatible type provided: {}'
                        .format(k, type(a[k]), type(v)))
            except AttributeError:
                # something else?
                pass
            else:
                # OK, move on.
                continue

            # give up.
            raise TypeError('Cannot merge key "{}", unsupported type(s): {}, {}'
                    .format(k, type(a[k]), type(v)))
        else:
            a[k] = v


class Graph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        self.experiment = None

        ## including this causes major problems with the to_undirected()
        ## function ... not worth it.
        #self.experiment = kwargs.get('experiment', None)

        super(Graph, self).__init__(*args, **kwargs)
        self._whereis_data = {}
        self._collision_nodes = {}
        self._collision_blobs = {}
        self._gap_nodes = []
        self._gap_blobs = []

        self.tag_edges()

    def tag_edges(self):
        """
        Record all original edges as the node ids can become totally
        unhinged (and are not even guaranteed to always be the same) from
        the original blob data. This information will be carried through
        condensation operations.
        """
        for a, b in self.edges_iter():
            self.edge[a][b][kc.BLOB_ID_EDGES] = {(a, b)}

    def copy(self):
        return type(self)(self, experiment=self.experiment)

    def components(self, node):
        return set(int(n) for n in self.node[node].get(kc.COMPONENTS, [node]))

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

        all_nodes = (node,) + other_nodes
        subg = self.subgraph(all_nodes)
        if nx.number_connected_components(subg.to_undirected()) != 1:
            raise ValueError('Attempting to merge unconnected nodes.')

        # not sure which function is trying to merge a node with itself...
        # but it needs to be stopped. until then catch it here.
        if node in other_nodes:
           other_nodes = list(other_nodes)
           other_nodes.pop(other_nodes.index(node))

        if not other_nodes:
            return

        edges_out = set(self.out_edges_iter(all_nodes))
        edges_in = set(self.in_edges_iter(all_nodes))
        #edges_internal = edges_out & edges_in
        edges_external = edges_out ^ edges_in

        if kc.COMPONENTS not in nd:
            nd[kc.COMPONENTS] = set([node])

        # copy/update node data (NO TOPOLOGY CHANGES)
        for other_node in other_nodes:
            other_data = self.node[other_node]

            # abscond with born/died
            if not skip_life_recalc:
                nd[kc.FRAME_BORN] = min(nd[kc.FRAME_BORN], other_data.pop(kc.FRAME_BORN))
                nd[kc.FRAME_DIED] = max(nd[kc.FRAME_DIED], other_data.pop(kc.FRAME_DIED))

                nd[kc.TIME_BORN] = min(nd[kc.TIME_BORN], other_data.pop(kc.TIME_BORN))
                nd[kc.TIME_DIED] = max(nd[kc.TIME_DIED], other_data.pop(kc.TIME_DIED))
            else:
                for key in (kc.FRAME_BORN, kc.FRAME_DIED, kc.TIME_BORN, kc.TIME_DIED):
                    del other_data[key]

            # combine set/mapping data
            nd[kc.COMPONENTS].update(
                    other_data.pop(kc.COMPONENTS, set([other_node])))

            merge_mappings(nd, other_data)
            # for k, v in six.iteritems(other_data):
            #     if k in nd:
            #         # works for dicts and sets.
            #         try:
            #             nd[k].update(v)
            #         except TypeError:
            #             pass
            #     else:
            #         nd[k] = v

        # propogate original edge data
        for a, b in edges_external:
            # should be in one and only one (xor)
            assert (a in all_nodes) ^ (b in all_nodes)

            if a in all_nodes:
                # "leaving" edge
                u = node
                v = b
            else:
                # "incoming" edge
                u = a
                v = node

            edge_data = self.get_edge_data(a, b)
            if self.has_edge(u, v):
                existing_edge_data = self.get_edge_data(u, v)
                merge_mappings(existing_edge_data, edge_data)
            else:
                self.add_edge(u, v, **edge_data)

        # cleanup
        for other_node in other_nodes:
            # remove nodes (edges come off with)
            self.remove_node(other_node)

        # update what's where
        for component in nd[kc.COMPONENTS]:
            self._whereis_data[component] = node

    def validate(self, acceptable_f_delta=-10):
        """
        Verify that the graph:

        * contains the requisite data attached to each node
        * contains only causal links
        """
        for node, node_data in self.nodes_iter(data=True):
            for req_key in [kc.FRAME_BORN, kc.FRAME_DIED, kc.TIME_BORN, kc.TIME_DIED]:
                if req_key not in node_data:
                    raise AssertionError("Node {} missing required key '{}'".format(
                            node, req_key))

        for a, b in self.edges_iter():
            f_delta = self.node[b][kc.FRAME_BORN] - self.node[a][kc.FRAME_DIED]
            if f_delta < acceptable_f_delta:
                raise AssertionError("Edge from {} to {} is acausal, going "
                        "back in time {:0.0f} frames".format(a, b, -f_delta))

        L.warn('Validation pass')

    def lifespan_f(self, node):
        # +1 because something that was born & died on the same frame exists
        # for 1 frame.
        return self.node[node][kc.FRAME_DIED] - self.node[node][kc.FRAME_BORN] + 1

    def lifespan_t(self, node):
        # This is off by one frame compared to the frame-based lifespan
        # because we don't know for sure how long the frame lasted
        return self.node[node][kc.TIME_DIED] - self.node[node][kc.TIME_BORN]

    def count_worms(self, experiment):
        worm_count_dict = network_number_wizard(self, experiment)
        for node, count in six.iteritems(worm_count_dict):
            self.node[node]['worm_count'] = count
            #print(self.node[node])

    def giant(self):
        """Return a subgraph copy of the giant component"""
        giant = sorted(nx.connected_component_subgraphs(self.to_undirected()),
                       key=len, reverse=True)[0]
        return giant

    def determine_moving(self, experiment):
        # code following network_number_wizard logic for determining
        # seeds
        terminals_df = experiment.prepdata.load('terminals')
        moving = []
        for node in self.nodes():
            is_moving = 0
            if node_is_moving(node, terminals_df):
                is_moving = 1
                moving.append(node)
            self.node[node]['moved'] = is_moving
        return moving

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

    def bridge_gaps(self, node_list, blob_list):
        self._gap_nodes.extend(node_list)
        self._gap_blobs.extend(blob_list)
        self.add_edges_from(node_list, taped=True)

    def untangle_collision(self, collision_node, collision_result,
                           blobs=None):
        """
        this untangles collisions by removing the collision
        nodes and merging the parent-child pairs that belong
        to the same worm.


        A   C
         \ /              A     C
       collision    ==>   | and |
         / \              B     D
        B   D

        The collision node and all nodes in collision result are
        removed from the self and replaced by compound nodes.


        params
        --------
        collision_node: (int or tuple)
           the node id (in self) that identifies the collision.
        collision_result: (list)
           Values are lists of node pairs to be joined.

           example:
           [[node_A, node_B], [node_C, node_D]]
        """

        if collision_result:
            self.remove_node(collision_node)

        for n1, n2 in collision_result:
            #parents = set(self.predecessors(n1))
            #children = set(self.successors(n2))

            # combine data
            #new_node, new_node_data = self.condense_nodes(n1, n2)
            try:
                n1_data = self.node[n1]
            except KeyError:
                print('Warning: {n} not found in graph'.format(n=n1))
                continue
            if 'collision' not in n1_data:
                self.node[n1]['collisions'] = set()
            self.node[n1]['collisions'].add(collision_node)
            self.add_edge(n1, n2)

        # document collision
        self._collision_nodes[collision_node] = collision_result
        self._collision_blobs[collision_node] = blobs


    def consolidate_node_data(self, experiment, node):
        """
        Returns a pandas DataFrame with all blob data for a node.
        Accepts both compound and single nodes.
        For compound nodes, data includes all components.

        params
        -----
        graph: (waldo.network.Graph object)
           a directed graph of node interactions
        experiment: (multiworm experiment object)
           the experiment from which data can be exctracted.
        node: (int or tuple)
           the id (from graph) for a node.

        returns
        -----
        all_data: (pandas DataFrame)
           index is 'frame'
           columns are:
           'area', 'centroid'
           'contour_encode_len', 'contour_encoded', 'contour_start',
           'midline', 'size', 'std_ortho', 'std_vector', 'time'
        """
        data = []
        for subnode in self.components(node):
            try:
                blob = experiment[subnode]
                df = blob.df
            except (KeyError, multiworm.MWTDataError) as e:
                print('{e} reading blob {i}'.format(e=e, i=subnode))
                continue

            if blob.empty:
                #print(subnode, 'subnode of', node, 'is empty')
                continue

            df.set_index('frame', inplace=True)
            df['blob'] = subnode
            data.append(df)
        if data:
            all_data = pd.concat(data)
            all_data.sort(inplace=True)
            return all_data

    def compound_bl_filter(self, experiment, threshold):
        """
        Return node IDs from *graph* and *experiment* if they moved at least
        *threshold* standard body lengths.
        """
        cbounds = self.node_movement(experiment)
        moved = cbounds[cbounds['bl'] >= threshold]
        return moved['bid'] if 'bid' in moved.columns else moved.index

    def node_movement(self, experiment):
        cbounds = self._compound_bounding_box(experiment)
        cbounds['bl'] = ((cbounds['x_max'] - cbounds['x_min'] +
                          cbounds['y_max'] - cbounds['y_min']) /
                          experiment.typical_bodylength)
        return cbounds

    def node_summary(self, experiment, node_ids=[]):
        """
        returns a dataframe with lots of data for every node specified in node_ids.

        dataframe columns are:
        'bl', 'components', 'f0', 'fN', 't0', 'tN',
        'x_max', 'x_min', 'y_max', y_min'

        params
        -----
        experiment: (wio.experiment object)
        node_ids: (list)

        """
        if not node_ids:
            node_ids = self.nodes(data=False)

        frame_times = experiment.frame_times
        node_movement = self.node_movement(experiment)
        node_movement.sort(inplace=True)
        node_summaries = []
        for node in node_ids:
            node_data = self.node[node]
            bf, df = node_data[kc.FRAME_BORN], node_data[kc.FRAME_DIED]
            t0 = frame_times[bf - 1]
            tN = frame_times[df - 1]
            comps = list(node_data.get('components', [node]))
            comp_string = '-'.join([str(c) for c in comps])
            n_summary = {'bid':node, 'f0':bf, 'fN':df, 't0':t0, 'tN':tN, 'components':comp_string}
            if node in node_movement.index:
                n_move = node_movement.loc[node]
                for i, v in zip(n_move.index, n_move):
                    n_summary[str(i)] = v
            else:
                for i in ['x_max', 'x_min', 'y_max', 'y_min', 'bl']:
                    n_summary[i] = 0
            node_summaries.append(n_summary)
        node_info = pd.DataFrame(node_summaries)
        node_info.set_index('bid', inplace=True)
        node_info.sort(inplace=True)
        return node_info

    def _compound_bounding_box(self, experiment):
        """
        Construct bounding box for all nodes (compound or otherwise) by using
        experiment prepdata and graph node components.
        """
        bounds = experiment.prepdata.bounds
        def wh(row):
            return self.where_is(row['bid'])

        bounds['node'] = bounds.apply(wh, axis=1)
        groups = bounds.groupby('node')
        bboxes = groups.agg({'x_min': min, 'x_max': max, 'y_min': min, 'y_max': max})
        bbox_nodes = set(bboxes.index)
        graph_nodes = set(self.nodes(data=False))
        actual_bbox_nodes = list(bbox_nodes & graph_nodes)
        return bboxes.loc[actual_bbox_nodes]


class MergableValueValueDict(dict):
    """
    NX adjacency format (e.g. Graph.adj or DiGraph.pred):
    {
        node1: {
            link1: { ... FFA bin of attributes ... },
            ...
        }
    }

    So...if we want to make attributes mergable, we want to monitor the
    values' (node data) values (link data).
    """
