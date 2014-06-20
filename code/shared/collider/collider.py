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

#import wio.file_manager as fm

__all__ = [
    'remove_fission_fusion',
    'remove_fission_fusion_rel',
    'remove_single_descendents',
    'remove_offshoots',
    'remove_nodes_outside_roi',
    'flat_node_list'
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
    for node in itertools.chain([start, end], others):
        try:
            components.update(graph.node[node]['components'])
        except KeyError:
            components.add(node)

    new_node_data = {
        'born': graph.node[start]['born'],
        'died': graph.node[end]['died'],
        'components': components,
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
