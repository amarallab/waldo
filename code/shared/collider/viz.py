# -*- coding: utf-8 -*-
"""
MWT collision graph visualizations
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

def render_nx_as_dot(nxgraph, output_file=None, format='png', colormap='jet_r', logrange=(-0.5, 4)):
    if output_file is None:
        output_file = 'graph.gv'

    # copy edges
    gvgraph = graphviz.Digraph(engine='dot', format=format)
    for edge in nxgraph.edges_iter():
        gvgraph.edge(*[str(node) for node in edge])

    # style nodes
    for node, node_data in nxgraph.nodes_iter(data=True):
        try:
            life = node_data['died'] - node_data['born']
            #if 100 <= node <= 110: print(life)
            if node_data.get('more', False):
                gvgraph.node(str(node), label='...', shape='circle', style='filled', color='grey')
            else:
                gvgraph.node(str(node), penwidth='5', color=get_color(life, logrange, colormap))
        except KeyError:
            pass

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

    return gvgraph.render(filename=output_file)

def direct_degree_distribution(digraph, maximums=(4, 4), flip_y=False, cmap=None):
    if cmap is None:
        cmap = plt.cm.Blues

    degrees = np.zeros(tuple(m+1 for m in maximums), dtype=int)
    for (in_node, in_deg), (out_node, out_deg) in zip(
            digraph.in_degree_iter(), digraph.out_degree_iter()):
        assert in_node == out_node # hopefully the iterators are matched...
        degrees[min(in_deg, degrees.shape[0]-1)][min(out_deg, degrees.shape[1]-1)] += 1

    f, ax = plt.subplots()
    heatmap = ax.pcolor(degrees.T, cmap=cmap)

    # http://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib
    for x in range(degrees.shape[0]):
        for y in range(degrees.shape[1]):
            deg = degrees[x,y]
            ax.text(x + 0.5, y + 0.5, deg, ha='center', va='center',
                    color='white' if deg > 0.6*np.max(degrees) else 'black')

    # http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
    ax.set_xticks(np.arange(degrees.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(degrees.shape[1])+0.5, minor=False)
    if flip_y:
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
    xticks, yticks = (list(range(t)) for t in degrees.shape)
    xticks[-1] = str(xticks[-1]) + '+'
    yticks[-1] = str(yticks[-1]) + '+'
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    ax.set_xlabel('In degree')
    ax.set_ylabel('Out degree')

    f.colorbar(heatmap)
    return f, ax
