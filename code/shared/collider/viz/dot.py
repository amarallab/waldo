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

def get_color(value, scalerange, cmap='jet'):
    cm = plt.cm.get_cmap(cmap)
    val = (np.log10(value+1e-100) - scalerange[0])/(scalerange[1] - scalerange[0])
    color = mplcolors.rgb2hex(cm(val)[:3])
    return color

def render_nx_as_dot(nxgraph, output_file=None, format='png',
                     colormap='jet_r', logrange=(-0.5, 4), ref=True,
                     focus=None):
    if output_file is None:
        output_file = 'graph.gv'

    # copy edges
    gvgraph = graphviz.Digraph(engine='dot', format=format)
    for edge in nxgraph.edges_iter():
        gvgraph.edge(*[str(node) for node in edge])

    # style nodes
    for node, node_data in nxgraph.nodes_iter(data=True):
        try:
            life = nxgraph.lifespan_f(node)
            #if 100 <= node <= 110: print(life)
            if node_data.get('more', False):
                gvgraph.node(str(node), label='...', shape='circle', style='filled', color='grey')
            else:
                gvgraph.node(str(node), penwidth='5', color=get_color(life, logrange, colormap))
        except KeyError:
            pass

    if ref:
        # reference
        reference = [0, 1, 3, 10, 30, 100, 300, 1000, 3000]
        ref_nodes = ['{} frames'.format(r) for r in reference]
        iterref = iter(ref_nodes)
        source = six.next(iterref)
        for dest in iterref:
            gvgraph.edge(source, dest)
            source = dest
        for ref, ref_node in zip(reference, ref_nodes):
            gvgraph.node(ref_node, penwidth='5', color=get_color(ref, logrange, colormap))

    if focus:
        gvgraph.node(str(focus), penwidth='2.5', shape='doubleoctagon')

    return gvgraph.render(filename=output_file)
