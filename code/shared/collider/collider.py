# -*- coding: utf-8 -*-
"""
MWT collision graph manipulations
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools
import collections
try:
    from statistics import mean # stdlib 3.4+
except ImportError:
    def mean(x):
        return sum(x) / len(x)

import numpy as np
import networkx as nx
import pandas as pd

import encoding.decode_outline as de
from images.manipulations import points_to_aligned_matrix

__all__ = [
    'remove_fission_fusion',
    'remove_fission_fusion_rel',
    'remove_single_descendents',
    'remove_offshoots',
    'remove_nodes_outside_roi',
    'flat_node_list',
    'resolve_collisions'
]

def _check_assumptions(graph):
    for node in graph:
        successors = graph.successors(node)
        predecessors = graph.predecessors(node)
        if len(successors) == 2:
            for successor, successor_indeg in graph.in_degree_iter(successors):
                if successor_indeg != 1:
                    print(node, '-->', successors)
                    raise AssertionError("Fission children have unexpected number of parents (not 1)")
        elif len(successors) == 1:
            successor = successors[0]
            successor_indeg = graph.in_degree(successor)
            if successor_indeg != 2:
                print(node, '-->', successors, 'indeg={}'.format(successor_indeg))
                raise AssertionError("Fusion child has unexpected number of parents (not 2)")

        if len(predecessors) == 2:
            for predecessor, predecessor_indeg in graph.out_degree_iter(predecessors):
                assert predecessor_indeg == 1, "Fission parents have unexpected number of children (not 1)"
        elif len(predecessors) == 1:
            assert graph.out_degree(predecessors[0]) == 2, "Fusion parent has unexpected number of children (not 2)"

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, six.string_types):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def frame_filter(threshold):
    def conditional(graph, nodes):
        for node in nodes:
            if graph.node[node]['died'] - graph.node[node]['born'] > threshold:
                return False
        return True
    return conditional

def lifespan(node):
    return node['died'] - node['born']

def condense_nodes(graph, start, end, *others):
    """
    *start*, *end*, and *others* are all (key, value) pairs where the key is
    the node name and value is the node data in dictionary form.

    Required node dictionary keys:
      * born
      * died

    Optional node dictionary keys:
      * components
    """
    # come up with new node name
    if start != end:
        start_and_end = list(flatten([start, end]))
        new_node = start_and_end[0], start_and_end[-1]
    else:
        new_node = start

    # repackage data
    components = set()
    collisions = set()
    for node in itertools.chain([start, end], others):
        try:
            components.update(graph.node[node]['components'])
        except KeyError:
            components.add(node)
        try:
            collisions.update(graph.node[node]['collisions'])
        except KeyError:
            pass

    new_node_data = {
        'born': graph.node[start]['born'],
        'died': graph.node[end]['died'],
        'components': components,
        'collisions': collisions,
    }

    return new_node, new_node_data

def remove_single_descendents(graph):
    """
    Combine direct descendents (and repetitions thereof) into a single node.

    ..
          ~~~~~
           \|/
            A           ~~~~~
            |            \|/
            |     ==>    A-B
            B            /|\
           /|\          ~~~~~
          ~~~~~

    The hidden data will be attached to the nodes as a set, for example from
    above: ``{A, B}``.
    """
    all_nodes = graph.nodes()

    while all_nodes:
        node = all_nodes.pop()
        if node not in graph:
            continue # node was already removed/abridged

        children = set(graph.successors(node))
        if len(children) != 1:
            continue
        child = children.pop()

        if len(graph.predecessors(child)) != 1:
            continue

        parents = set(graph.predecessors(node))
        grandchildren = set(graph.successors(child))

        new_node, new_node_data = condense_nodes(graph, node, child)

        graph.add_node(new_node, **new_node_data)
        graph.add_edges_from((p, new_node) for p in parents)
        graph.add_edges_from((new_node, gc) for gc in grandchildren)
        graph.remove_nodes_from([node, child])

        all_nodes.append(new_node)

    # graph is modified in-place

def remove_fission_fusion(graph, max_split_frames=None):
    """
    Strip out fission-fusion events (and repetitions thereof) from the
    graph.

    ..
           ~~~
            |
            A
           / \           ~~~
          /   \           |
         B     C   ==>   A-D
          \   /           |
           \ /           ~~~
            D
            |
           ~~~

    The hidden data will be attached to the nodes as a set, for example from
    above: ``{A, B, C, D}``.

    """
    if max_split_frames is None:
        conditional = None
    else:
        conditional = frame_filter(max_split_frames)

    all_nodes = graph.nodes()

    while all_nodes:
        node = all_nodes.pop()
        if node not in graph:
            continue # node was already removed/abridged

        parents = set(graph.predecessors(node))
        children = set(graph.successors(node))

        if len(children) != 2:
            continue # no fission occured
            # (probably a job for remove_single_descendents())

        grandchildren = set()
        abort = False
        for child in children:
            new_gc = graph.successors(child)
            if len(new_gc) != 1:
                abort = True
                break
            grandchildren.update(new_gc)
        if abort:
            continue # see TestFissionFusion.test_child_swap

        if len(grandchildren) != 1:
            continue # skip doing anything

        if conditional is not None:
            if not conditional(graph, children):
                continue # children fail conditional testing

        grandchild = grandchildren.pop()
        greatgrandchildren = set(graph.successors(grandchild))

        new_node, new_node_data = condense_nodes(graph, node, grandchild, *children)

        graph.add_node(new_node, **new_node_data)
        graph.add_edges_from((p, new_node) for p in parents)
        graph.add_edges_from((new_node, ggc) for ggc in greatgrandchildren)

        graph.remove_node(node)
        graph.remove_nodes_from(children)
        graph.remove_node(grandchild)

        all_nodes.append(new_node)

    # graph is modified in-place

def remove_fission_fusion_rel(digraph, split_rel_time):
    """
    Strip out fission-fusion events (and repetitions thereof) from the
    digraph.

    ..
           ~~~
            |
            A
           / \           ~~~
          /   \           |
         B     C   ==>   A-D
          \   /           |
           \ /           ~~~
            D
            |
           ~~~

    The hidden data will be attached to the nodes as a set, for example from
    above: ``{A, B, C, D}``.

    """
    def conditional(digraph, node, children, grandchild):
        # average age of focal node/gchild
        endpoint_avg = mean([lifespan(digraph.node[node]),
                        lifespan(digraph.node[grandchild])])
        # average age of children
        split_avg = mean([lifespan(digraph.node[c]) for c in children])

        return split_avg / endpoint_avg <= split_rel_time

    all_nodes = digraph.nodes()
    all_nodes.sort() # order matters here, random key hashes...yadda yadda.

    while all_nodes:
        node = all_nodes.pop() # pop latest node, work end to start
        if node not in digraph:
            continue # node was already removed/abridged

        parents = set(digraph.predecessors(node))
        children = set(digraph.successors(node))

        if len(children) != 2:
            continue # no fission occured
            # (probably a job for remove_single_descendents())

        grandchildren = set()
        abort = False
        for child in children:
            new_gc = digraph.successors(child)
            if len(new_gc) != 1:
                abort = True
                break
            grandchildren.update(new_gc)
        if abort:
            continue # see TestFissionFusion.test_child_swap

        if len(grandchildren) != 1:
            continue # skip doing anything

        grandchild = grandchildren.pop()

        if not conditional(digraph, node, children, grandchild):
            continue # failed conditional testing

        greatgrandchildren = set(digraph.successors(grandchild))

        new_node, new_node_data = condense_nodes(digraph, node, grandchild, *children)

        digraph.add_node(new_node, **new_node_data)
        digraph.add_edges_from((p, new_node) for p in parents)
        digraph.add_edges_from((new_node, ggc) for ggc in greatgrandchildren)

        digraph.remove_node(node)
        digraph.remove_nodes_from(children)
        digraph.remove_node(grandchild)

        all_nodes.append(new_node)

    # graph is modified in-place

def remove_nodes_outside_roi(graph, experiment, x, y, r):
    """
    Removes nodes that are outside of a precalculated
    circle or 'region of interest'.  Must run before other simplifications;
    does not tolerate compound blob IDs

    params
    -----
    graph: (networkx graph)
       nodes are blob ids
    experiment: (multiworm Experiment)
       the experiment object corresponding to the same recording
    ex_id: (str)
       the experiment id (ie. timestamp) used to look up roi.
    """
    def box_centers(experiment):
        bids, boxes = [], []
        for (bid, blob_data) in experiment.all_blobs():
            if not blob_data:
                continue
            if 'centroid' in blob_data:
                xy = blob_data['centroid']
                #print(bid, len(xy))
                if xy != None and len(xy) > 0:
                    x, y = zip(*xy)
                    xmin, xmax = min(x), max(x)
                    ymin, ymax = min(y), max(y)
                    box = [xmin, ymin, xmax, ymax]
                    bids.append(bid)
                    boxes.append(box)

        xmin, ymin, xmax, ymax = zip(*boxes)
        box_centers = np.zeros((len(boxes), 2), dtype=float)
        box_centers[:, 0] = (np.array(xmin) + np.array(xmax)) / 2
        box_centers[:, 1] = (np.array(ymin) + np.array(ymax)) / 2
        return bids, box_centers

    #calculate
    bids, box_centers = box_centers(experiment)
    dists = np.sqrt((box_centers[:, 0] - x)**2 +
                   (box_centers[:, 1] - y)**2)

    are_inside = dists < r

    outside_nodes = []
    for bid, in_roi in zip(bids, are_inside):
        if not in_roi:
            outside_nodes.append(bid)

    graph.remove_nodes_from(outside_nodes)

def remove_offshoots(digraph, threshold):
    """
    Remove small dead-ends from *digraph* that last less than *threshold*
    frames.
    """
    all_nodes = digraph.nodes()
    filt = frame_filter(threshold)

    while all_nodes:
        node = all_nodes.pop()
        if digraph.in_degree(node) != 1 or digraph.out_degree(node) != 0:
            continue # topology wrong

        if not filt(digraph, [node]):
            continue # lasts too long

        # add to components of parent then remove node
        parent = digraph.predecessors(node)[0]

        _, new_node_data = condense_nodes(digraph, parent, parent, node)
        digraph.node[parent] = new_node_data

        digraph.remove_node(node)

def flat_node_list(graph):
    """
    returns list of all non-compund nodes in a graph, inculding
    all the nodes that are inside compound nodes.
    """
    node_ids = []
    for node in graph.nodes():
        if type(node) != tuple:
            node_ids.append(node)
        else:
            node_data = graph.node[node]
            if 'components' in node_data:
                internal_nodes = list(node)
            else:
                internal_nodes = list(node)
            node_ids.extend(internal_nodes)
    return list(set(node_ids))

def grab_outline(node, experiment, first=True):
    """
    return the first or last complete outline for a given node
    as a list of points.

    params
    -----
    node: (int or tuple)
       the id (from graph) for a node.
    experiment: (multiworm experiment object)
       the experiment from which data can be exctracted.
    first: (bool)
       toggle that deterimines if first or last outline is returned

    returns
    ----
    outline: (list of tuples)
       a list of (x,y) points
    """

    #print('grabbing', node)
    i =0
    go_backwards = False
    if not first:
        i = -1
        go_backwards = True

    if type(node) == tuple:
        node = node[i]
    if type(node) == str and '-' in node:
        node = node.split('-')[i]

    data  = experiment.parse_blob(node)

    x, y = zip(*data['contour_start'])
    contour_encode_len = data['contour_encode_len']
    contour_encoded = data['contour_encoded']
    encoded_outline = zip(x, y, contour_encode_len,contour_encoded)

    if go_backwards:
        # needs to be a list, not an iterable if I want to go backwards
        encoded_outline = list(encoded_outline)
        encoded_outline.reverse()

    for i, o in enumerate(encoded_outline):
        if not o:
            continue  #to avoid None from breaking the loop
        if o[2] != None:
            outline_points = de.decode_outline(o)
            #print(outline_points)
            return outline_points
    else:
        print('Failed to find outline')
        print('grabbing', node, type(node))


# TODO make this function agnostic to the number of parents.
def create_collision_masks(graph, experiment, node, verbose=False):
    """
    return lists of (node_id, binary mask) tuples for
    the parents and children of the specified node.

    binary masks are boolean np.arrays containing the filled area
    of the worm shape. all masks are in the same, reduced
    coordinate system.

    params
    -----
    graph: (networkx graph object)
       a directed graph of node interactions
    experiment: (multiworm experiment object)
       the experiment object cooresponding to the same nodes
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
    p = list(set(graph.predecessors(node)))
    c = list(set(graph.successors(node)))
    #grab relevant outlines.
    p0 = grab_outline(p[0], experiment, first=False)
    p1 = grab_outline(p[1], experiment, first=False)
    #m0 = grab_outline(node, experiment, first=True)
    #m1 = grab_outline(node, experiment, first=False)
    c0 = grab_outline(c[0], experiment, first=True)
    c1 = grab_outline(c[1], experiment, first=True)
    if verbose:
        print('parents:{p}'.format(p=p))
        print('children:{c}'.format(c=c))
        #print('beginning:end pixel overlap')

    # align outline masks
    outline_list = [p0, p1, c0, c1]
    masks, bbox = points_to_aligned_matrix(outline_list)
    #reorganize
    parents = [i for i in zip(p, masks[:2])]
    children = [i for i in zip(c, masks[-2:])]
    return parents, children


#TODO: make agnostic to number of parents/children
def compare_masks(parents, children, err_margin=10, verbose=True):
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
    err = 'parents and children must be lists with two elements'
    assert len(parents) == len(children) == 2, err
    d1_sum, d2_sum= 0, 0
    d1, d2 = [], []
    p_nodes, _ = zip(*parents)
    c_nodes, _ = zip(*children)
    comparison = pd.DataFrame(index=p_nodes, columns=c_nodes)
    for i, (p, p_mask) in enumerate(parents):
        for j, (c, c_mask) in enumerate(children):
            overlap = (p_mask & c_mask).sum()
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

def resolve_collisions(graph, experiment, collision_nodes):
    """

    Removes all the collisions that can be resolved
    through pixel-overlap from the graph.
    If a collision cannot be resolved, it remains in the graph.

    only works if collisions nodes have two parents and two children.

    params
    -----
    graph: (networkx graph object)
       a directed graph of node interactions
    experiment: (multiworm experiment object)
       the experiment object cooresponding to the same nodes
    collision_nodes: (list)
       a list of nodes that are suspected to be collisions between
       worms.
    """
    collision_results = {}
    for node in collision_nodes:
        parent_masks, children_masks = create_collision_masks(graph, experiment, node)
        collision_result = compare_masks(parent_masks, children_masks)
        #print(node, 'is', collision_result)
        if len(collision_result):
            collision_results[node] = collision_result
            untangle_collision(graph, node, collision_result)


def untangle_collision(graph, collision_node, collision_result):
    """
    this untangles collisions by removing the collision
    nodes and merging the parent-child pairs that belong
    to the same worm.


     A   C
      \ /
   collision    ==>  (A, B) and (C, D)
     / \
    B   D

    The collision node and all nodes in collision result are
    removed from the graph and replaced by compound nodes.


    params
    --------
    graph: (networkx graph object)
    collision_node: (int or tuple)
       the node id (in graph) that identifies the collision.
    collision_result: (list)
       Values are lists of node pairs to be joined.

       example:
       [[node_A, node_B], [node_C, node_D]]
    """

    if len(collision_result):
        graph.remove_node(collision_node)
    #print(col)
    for cr in collision_result:
        #print(cr)
        n1, n2 = cr

        parents = set(graph.predecessors(n1))
        children = set(graph.successors(n2))

        # combine data
        new_node, new_node_data = condense_nodes(graph, n1, n2)
        if 'collision' not in new_node_data:
            new_node_data['collisions'] = set()
        new_node_data['collisions'].add(collision_node)

        # add merged node and link to previous parents/children
        graph.add_node(new_node, **new_node_data)
        graph.add_edges_from((p, new_node) for p in parents)
        graph.add_edges_from((c, new_node) for c in children)

        # remove old nodes.
        graph.remove_node(n1)
        graph.remove_node(n2)
