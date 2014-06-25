# -*- coding: utf-8 -*-
"""
MWT collision graph visualizations - Degree order
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def direct_degree_distribution(digraph, maximums=(4, 4), flip_y=False, cmap='Blues', nonzero=False):
    cmap = plt.get_cmap(cmap)

    degrees = np.zeros(tuple(m+1 for m in maximums), dtype=int)
    for (in_node, in_deg), (out_node, out_deg) in zip(
            digraph.in_degree_iter(), digraph.out_degree_iter()):
        assert in_node == out_node # hopefully the iterators are matched...
        degrees[min(in_deg, degrees.shape[0]-1)][min(out_deg, degrees.shape[1]-1)] += 1

    f, ax = plt.subplots()
    heatmap = ax.pcolor(degrees.T, cmap=cmap)

    if nonzero:
        degrees[0][0] = 0

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
