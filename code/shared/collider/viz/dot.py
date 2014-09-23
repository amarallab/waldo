# -*- coding: utf-8 -*-
"""
MWT collision graph visualizations - Display a dot graph via Graphviz
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import networkx as nx

import graphviz

GRAPHVIZ_ATTRIBUTES = ['shape', 'label', 'style',
                       'penwidth', 'color', 'fill']

def render_nx_as_dot(nxgraph, output_file=None, format='png',
                     colormap='jet_r', ref=None):
    """


    """
    if output_file is None:
        output_file = 'graph.gv'

    # copy edges
    gvgraph = graphviz.Digraph(engine='dot', format=format)
    for edge in nxgraph.edges_iter():
        gvgraph.edge(*[str(node) for node in edge])

    # style nodes
    unsafe_attributes = []
    for node, node_data in nxgraph.nodes_iter(data=True):
        safe_ones = {}
        for attribute, value in node_data.iteritems():
            #print(attribute, value)
            if attribute in GRAPHVIZ_ATTRIBUTES:
                safe_ones[attribute] =  value
            else:
                unsafe_attributes.append(attribute)
        #print(node, safe_ones)
        gvgraph.node(str(node), **safe_ones)
    print('tried to use the following non graphviz attributes:')
    print(set(unsafe_attributes))
    return gvgraph.render(filename=output_file)

def clear_formatting(nxgraph):
    for node, node_data in nxgraph.nodes_iter(data=True):
        for attr in GRAPHVIZ_ATTRIBUTES:
            if attr in node_data:
                del node_data[attr]

def format_graph_for_lifespan(nxgraph, focus=None, ref=False, cmap='jet'):
    cm = plt.cm.get_cmap(cmap)
    def lifespan_to_color(lifespan, scalerange=(-0.5, 4), cm=cm):
        val = (np.log10(lifespan+1e-100) - scalerange[0])/(scalerange[1] - scalerange[0])
        color = mplcolors.rgb2hex(cm(val)[:3])
        return color

    for node, node_data in nxgraph.nodes_iter(data=True):
        lifespan = nxgraph.lifespan_f(node)
        node_data['color'] = lifespan_to_color(lifespan)
        node_data['shape'] = 'ellipse'

        if node_data.get('more', False):
            node_data['label'] = '...'
            node_data['shape'] = 'circle'
            node_data['style'] = 'filled'
            node_data['color'] = 'grey'

        if node == focus:
            node_data['penwidth'] = '2.5'
            node_data['shape'] = 'doubleoctagon'

    print(nxgraph)
    if ref:
        # reference
        reference = [0, 1, 3, 10, 30, 100, 300, 1000, 3000]
        ref_nodes = ['{} frames'.format(r) for r in reference]
        iterref = iter(ref_nodes)
        source = six.next(iterref)
        for dest in iterref:
            nxgraph.add_edge(source, dest)
            source = dest
        for ref, ref_node in zip(reference, ref_nodes):
            nxgraph.add_node(ref_node, penwidth='5',
                             color=lifespan_to_color(ref))

def format_graph_for_moved(nxgraph, cmap='jet', ref=False):
    cm = plt.cm.get_cmap(cmap)

    def bin_to_color(wc):
        wc = int(wc) / 3.0
        color = mplcolors.rgb2hex(cm(wc)[:3])
        return color

    for node, node_data in nxgraph.nodes_iter(data=True):
        m = node_data.get('moved', None)
        if m is None:
            continue

        m = node_data['moved']
        node_data['color'] = bin_to_color(m)
        node_data['shape'] = 'ellipse'
        if m:
            node_data['style'] = 'filled'

        if node_data.get('more', False):
            node_data['label'] = '...'
            node_data['shape'] = 'circle'
            node_data['style'] = 'filled'
            node_data['color'] = 'grey'

    if ref:
        nxgraph.add_edge('moved < 10 pxl', 'moved >= 10pxl')
        nxgraph.add_node('moved < 10pxl', penwidth='5',
                         color=bin_to_color(ref))
        nxgraph.add_node('moved >= 10pxl', penwidth='5',
                         color=bin_to_color(ref),
                         style='filled')

def format_graph_for_worm_counts(nxgraph, cmap='jet', ref=False):

    cm = plt.cm.get_cmap(cmap)
    levels = [.5, 1, 2, 3]

    def wc_to_color(wc):
        wc = int(wc) / 3.0
        color = mplcolors.rgb2hex(cm(wc)[:3])
        return color

    for node, node_data in nxgraph.nodes_iter(data=True):
        worm_count = node_data.get('worm_count', None)
        if worm_count is None:
            continue

        node_data['color'] = wc_to_color(worm_count)
        node_data['shape'] = 'ellipse'
        node_data['style'] = 'filled'

        if node_data.get('more', False):
            node_data['label'] = '...'
            node_data['shape'] = 'circle'
            node_data['style'] = 'filled'
            node_data['color'] = 'grey'

    if ref:
        # reference
        reference = levels
        ref_nodes = ['{} worms'.format(r) for r in reference]
        iterref = iter(ref_nodes)
        source = six.next(iterref)
        for dest in iterref:
            nxgraph.add_edge(source, dest)
            source = dest
        for ref, ref_node in zip(reference, ref_nodes):
            nxgraph.add_node(ref_node, penwidth='5',
                             color=wc_to_color(ref),
                             style='filled')

def format_graph_for_true_counts(nxgraph, cmap='jet', ref=True):

    cm = plt.cm.get_cmap(cmap)
    levels = [.5, 1, 2, 3]

    def wc_to_color(wc):
        wc = int(wc) / 3.0
        color = mplcolors.rgb2hex(cm(wc)[:3])
        return color

    for node, node_data in nxgraph.nodes_iter(data=True):
        worm_count = node_data.get('true_count', 0)
        node_data['color'] = wc_to_color(worm_count)
        node_data['shape'] = 'ellipse'

        if worm_count:
            node_data['style'] = 'filled'

        if node_data.get('more', False):
            node_data['label'] = '...'
            node_data['shape'] = 'circle'
            node_data['style'] = 'filled'
            node_data['color'] = 'grey'

    if ref:
        # reference
        reference = levels
        ref_nodes = ['{} worms'.format(r) for r in reference]
        ref_nodes = ['unknown'] + ref_nodes
        iterref = iter(ref_nodes)
        source = six.next(iterref)
        for dest in iterref:
            nxgraph.add_edge(source, dest)
            source = dest

            nxgraph.add_node('unknown', penwidth='5', color=wc_to_color(0))

        for ref, ref_node in zip(reference, ref_nodes[1:]):
            nxgraph.add_node(ref_node, penwidth='5',
                             color=wc_to_color(ref),
                             style='filled')

def format_graph_for_id(nxgraph, cmap='jet'):
    cm = plt.cm.get_cmap(cmap)

    def bin_to_color(wc):
        color = mplcolors.rgb2hex(cm(wc)[:3])
        return color

    for node, node_data in nxgraph.nodes_iter(data=True):
        m = node_data.get('moved', None)
        if m is None:
            continue
        node_data['color'] = bin_to_color(m)
        node_data['shape'] = 'ellipse'
        if m:
            node_data['style'] = 'filled'

        if node_data.get('more', False):
            node_data['label'] = '...'
            node_data['shape'] = 'circle'
            node_data['style'] = 'filled'
            node_data['color'] = 'grey'
