from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

from IPython.core.display import Image as display_image

from .dot import (
    render_nx_as_dot, clear_formatting,
    format_graph_for_lifespan,
    format_graph_for_worm_counts,
    format_graph_for_true_counts,
    format_graph_for_moved
)

from ..subgraph import nearby, neartime

def look(graph, target, jumps, ref=False, ctype='lifespan'):
    """
    In *graph*, a waldo.network.Graph, around *target*, show the network out to
    an (undirected) distance of *jumps*.  Optionally show a colored reference.
    """
    subgraph = nearby(graph, target, jumps)
    if ctype == 'lifespan':
        format_graph_for_lifespan(subgraph, ref=ref, focus=graph.where_is(target))
    elif ctype == 'worm_count':
        format_graph_for_worm_counts(subgraph, ref=ref)
    elif ctype == 'true_count':
        format_graph_for_true_counts(subgraph, ref=ref)
    elif ctype == 'moved_bool':
        format_graph_for_moved(subgraph, ref=ref)

    temp_file = render_nx_as_dot(subgraph)
    return display_image(temp_file)

def save_graphs(ex_id, graph, target, jumps, ref=False):
    """
    In *graph*, a waldo.network.Graph, around *target*, show the network out to
    an (undirected) distance of *jumps*.  Optionally show a colored reference.
    """



    subgraph = nearby(graph, target, jumps)


    format_graph_for_lifespan(subgraph, focus=graph.where_is(target))
    of = '{eid}_lifespan.gv'.format(eid=ex_id)
    print(of)
    temp_file = render_nx_as_dot(subgraph, output_file=of)
    clear_formatting(subgraph)

    format_graph_for_worm_counts(subgraph)
    of = '{eid}_worm_counts.gv'.format(eid=ex_id)
    temp_file = render_nx_as_dot(subgraph, output_file=of)
    clear_formatting(subgraph)

    format_graph_for_true_counts(subgraph)
    of = '{eid}_true_counts.gv'.format(eid=ex_id)
    temp_file = render_nx_as_dot(subgraph, output_file=of)
    clear_formatting(subgraph)

    format_graph_for_moved(subgraph)
    of = '{eid}_seed_counts.gv'.format(eid=ex_id)
    temp_file = render_nx_as_dot(subgraph, output_file=of)
    clear_formatting(subgraph)

    return display_image(temp_file)


def look2(graph, target, jumps, ref=False):
    """
    In *graph*, a waldo.network.Graph, around *target*, show the network out to
    an (undirected) distance of *jumps*.  Optionally show a colored reference.
    """
    subgraph = nearby(graph, target, jumps)
    format_graph_for_worm_counts(subgraph, ref=ref)
    temp_file = render_nx_as_dot(subgraph)
    return display_image(temp_file)

def look_time(graph, fstart, fend, ref=False):
    subgraph = neartime(graph, fstart, fend)
    temp_file = render_nx_as_dot(subgraph, ref=ref)
    return display_image(temp_file)
