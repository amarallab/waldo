from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

from IPython.core.display import Image as display_image

from .dot import render_nx_as_dot
from ..subgraph import nearby, neartime

def look(graph, target, jumps, ref=False):
    """
    In *graph*, a collider.Graph, around *target*, show the network out to
    an (undirected) distance of *jumps*.  Optionally show a colored reference.
    """
    subgraph = nearby(graph, target, jumps)
    temp_file = render_nx_as_dot(subgraph, ref=ref, focus=graph.where_is(target))
    return display_image(temp_file)

def look_time(graph, fstart, fend, ref=False):
    subgraph = neartime(graph, fstart, fend)
    temp_file = render_nx_as_dot(subgraph, ref=ref)
    return display_image(temp_file)
