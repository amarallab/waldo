# -*- coding: utf-8 -*-
"""
Manipulations removing degree-one things (short offshoots and basic chains)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library

# third party
import numpy as np
import pandas as pd

# package specific
#from waldo.conf import settings
from waldo.images.manipulations import points_to_aligned_matrix

from ..util import consolidate_node_data
import waldo.encoding.decode_outline as de


__all__ = [
    'CollisionException',
    'CollisionResolver',
]


class CollisionException(Exception):
    pass

class CollisionResolver(object):
    """
    Initalized with a wio.Experiment-like object and a simplified graph.
    """
    def __init__(self, experiment, graph):#, regenerate_cache=False):
        self._experiment = experiment
        self._graph = graph
        self._mask_counts = {}
        #self._terminals = self._experiment.prepdata.load('terminals').set_index('bid')
        self.reports = {}
        self.parent_node_to_blob = {}
        self.child_node_to_blob = {}

    def grab_outline(self, node, first=True, verbose=False):
        """
        return the first or last complete outline for a given node
        as a list of points.

        params
        -----
        node: (int or tuple)
           the id (from graph) for a node.
        graph: (networkx graph object)
           a directed graph of node interactions
        experiment: (multiworm experiment object)
           the experiment from which data can be exctracted.
        first: (bool)
           toggle that deterimines if first or last outline is returned

        returns
        ----
        outline: (list of tuples)
           a list of (x,y) points
        """
        graph = self._graph
        experiment = self._experiment
        nodes = [node]

        # older stuff that is suspected of being wrong.
        # preds = graph.predecessors(node)
        # while preds == 1:
        #     current = preds[0]
        #     nodes.insert(0, current)
        #     preds = graph.predecessors(current)

        # if first == True, we are dealing with a child of a collision.
        # then failure to locate an outline should search the successors.

        # invert this if first == False.

        if first:
            next_node = graph.successors
        else:
            next_node = graph.predecessors

        current = node
        backup_nodes = next_node(current)
        while len(backup_nodes) == 1:
            current=backup_nodes[0]
            nodes.append(current)
            backup_nodes = next_node(current)

        node_count = len(nodes)
        while len(nodes) > 0:
            node = nodes.pop(0)
            df = consolidate_node_data(graph, experiment, node)
            if df is None:
                print('Failed to find node data')
                #print('grabbing', node, type(node))
                raise CollisionException


            # if first ==True: sort chronologically. else sort in reverse.
            if first:
                df.sort(ascending=True, inplace=True)
            else:
                df.sort(ascending=False, inplace=True)

            for frame, row in df.iterrows():
                x, y = row['contour_start']
                l = row['contour_encode_len']
                enc = row['contour_encoded']
                bid = row['blob']
                is_good = True
                if not enc or not l:
                    is_good = False
                if not isinstance(enc, basestring):
                    is_good = False
                if is_good:
                    # record which blob id the outline originated from
                    if first:
                        # for children, we want first outline
                        self.child_node_to_blob[int(node)] = int(bid)
                    else:
                        # for parents, we want last outline
                        self.parent_node_to_blob[int(node)] = int(bid)
                    # return list of outline points
                    outline_points = de.decode_outline([x, y, l, enc])
                    return outline_points
        if verbose:
            print('I: Failed to find outline in %d predeccessors' % node_count)
            print('I: grabbing', node, type(node))
        return None


    def create_collision_masks(self, node, verbose=False):
        """
        return lists of (node_id, binary mask) tuples for
        the parents and children of the specified node.

        binary masks are boolean np.arrays containing the filled area
        of the worm shape. all masks are in the same, reduced
        coordinate system.

        params
        -----
        node: (int or tuple)
           the id (in graph) used to identify a collision
        verbose: (bool)
           a toggle to turn on/off print statements

        returns
        -----
        parents: (list of tuples)
           each element in parents is a tuple of (node_id, mask_for_node)
        children: (list of tuples)
           same structure as parents.

        """
        graph = self._graph

        p = list(set(graph.predecessors(node)))
        c = list(set(graph.successors(node)))
        if verbose:
            print('parents:{p}'.format(p=p))
            print('children:{c}'.format(c=c))
            #print('beginning:end pixel overlap')
        #grab relevant outlines.
        p0 = self.grab_outline(p[0], first=False)
        p1 = self.grab_outline(p[1], first=False)
        c0 = self.grab_outline(c[0], first=True)
        c1 = self.grab_outline(c[1], first=True)

        # align outline masks
        outline_list = [o for o in [p0, p1, c0, c1] if o is not None]
        #print(len(outline_list), 'outlines found for', node)
        if not outline_list:
            raise CollisionException
        masks, bbox = points_to_aligned_matrix(outline_list)

        next = 0

        if p0 is not None:
            p0 = masks[0]
            next += 1

        if p1 is not None:
            p1 = masks[next]
            next += 1

        if c0 is not None:
            c0 = masks[next]
            next += 1

        if c1 is not None:
            c1 = masks[next]

        parents = [(p[0], p0), (p[1], p1)]
        children = [(c[0], c0), (c[1], c1)]
        # document
        self._mask_counts[node] = len(outline_list)
        return parents, children

    def compare_masks(self, parents, children, err_margin=10, verbose=False):
        """
        returns a list of [parent, child] matches that maximize the
        amount of pixel overlap between parents and children.

        The winning matchup must have > err_margin pixels overlapping
        to be counted as a match. If this is not met, return empty list.

        For:
        parents:  A  B
                   \/
                   /\
        children  C  D

        the possible outcomes:
        (1) [[A, C], [B, D]]
        (2) [[A, D], [B, C]]
        (3) []
        if the number of overlapping pixes for (1) > (2) + err_margin,
        then (1) is the output.

        params
        -----
        parents: (list of tuples)
           Each element in parents is a (node_id, mask_for_node) tuple.
           This list must contain only two parents.
        children: (list of tuples)
           same structure as parents.
        err_margin: (int)
           number of pixels a match-up must exceed the other to be legit.
        verbose: (bool)
           a toggle to turn on/off print statements

        returns
        -----
        parent_child_matches: (list of lists)
           each element in the list contains a [parent, child] match as
           determined by the best set of overlapping
        """
        # TODO: remove assertation when number of comparisions irrelevant
        if len(parents) != 2 or len(children) != 2:
            raise ValueError('parents and children must be lists with two elements')

        NONE_OVERLAP_VALUE = 0
        d1_sum, d2_sum = 0, 0
        d1, d2 = [], []
        p_nodes, _ = zip(*parents)
        c_nodes, _ = zip(*children)
        comparison = pd.DataFrame(index=p_nodes, columns=c_nodes) if verbose else None
        for i, (p, p_mask) in enumerate(parents):
            for j, (c, c_mask) in enumerate(children):
                if p_mask is None or c_mask is None:
                    overlap = NONE_OVERLAP_VALUE
                else:
                    overlap = (p_mask & c_mask).sum()
                if verbose:
                    comparison.loc[p, c] = overlap
                if i == j:
                    d1_sum += overlap
                    d1.append([p, c])
                else:
                    d2_sum += overlap
                    d2.append([p, c])

        if verbose:
            print('prnts\tchildren')
            print(comparison)
            print('   diag \\   {s}   \t{d}'.format(s=d1_sum, d=d1))
            print('   diag /   {s}   \t{d}'.format(s=d2_sum, d=d2))

        if d1_sum > d2_sum + err_margin:
            return d1
        elif d2_sum > d1_sum + err_margin:
            return d2
        else:
            return []

    def result_nodes_to_blobs(self, collision_result):

        pnodes, cnodes = zip(*collision_result)
        # print(type(pnodes[0]))
        # print('parents')
        # print(self.parent_node_to_blob)
        # print( 'children')
        # print(self.child_node_to_blob)
        pblobs = [self.parent_node_to_blob.get(n, None) for n in pnodes]
        cblobs = [self.child_node_to_blob.get(n, None) for n in cnodes]
        return zip(pblobs, cblobs)

    def resolve_overlap_collisions(self, collision_nodes):
        """

        Removes all the collisions that can be resolved
        through pixel-overlap from the graph.
        If a collision cannot be resolved, it remains in the graph.

        only works if collisions nodes have two parents and two children.

        params
        -----
        collision_nodes: (list)
           a list of nodes that are suspected to be collisions between
           worms.
        """
        graph = self._graph

        collision_results = {}
        result_report = {'resolved': [], 'missing_data': [], 'no_overlap': [],
                         'collision_lifespans_t': {}}
        for node in collision_nodes:
            try:
                a = self.create_collision_masks(node)
                parent_masks, children_masks = a
                collision_result = self.compare_masks(parent_masks, children_masks)

                ####### UNCOMMENT to start resolving MULTI-COLLISIONS
                # a = generalized_create_collision_masks(graph, experiment, node,
                #                            report=True)
                # parent_masks, children_masks, report = a
                # collision_result = generalized_compare_masks(parent_masks, children_masks)
                ####################################################

                if self._mask_counts[node] < 4:
                    result_report['missing_data'].append(node)

                #INFO: if you want to see the outmasks and the relation between them
                # data = [parent_masks[0], parent_masks[1], children_masks[0], children_masks[1]]
                # outmasks = {i: o for i, o in data if o is not None}
                #
                # v = [len(c) for c in collision_result]
                # if len(v) == 0:
                #     print("Problems in collision_result: ", collision_result, " node: ", node)
                # else:
                #     cols = max(v)
                #     f, axs = plt.subplots(cols, len(collision_result))
                #     for i, cr in enumerate(collision_result):
                #         for col, id in enumerate(cr):
                #             axs[i][col].axis('off')
                #             if id in outmasks:
                #                 axs[i][col].imshow(outmasks[id], interpolation='none')
                #     plt.show()

            except CollisionException:
                #print('Warning: {n} has insuficient parent/child data to resolve collision'.format(n=node))
                result_report['missing_data'].append(node)
                continue
            #print(node, 'is', collision_result)
            if collision_result:
                result_report['collision_lifespans_t'][node] = graph.lifespan_t(node)
                collision_results[node] = collision_result
                #print('result', collision_result)
                blobs = self.result_nodes_to_blobs(collision_result)
                graph.untangle_collision(node, collision_result, blobs=blobs)
                result_report['resolved'].append(node)

                # temporary reporting to track down where long tracks dissapear to.
                long_collisions = [l for l in result_report['collision_lifespans_t'].values() if l > 60 * 20]
                if long_collisions:
                    print(len(long_collisions), 'collisions removed that were longer than 20min')
            else:
                result_report['no_overlap'].append(node)
        return result_report
