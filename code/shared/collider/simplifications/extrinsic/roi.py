# -*- coding: utf-8 -*-
"""
Manipulations removing degree-one things (short offshoots and basic chains)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np

from ..util import frame_filter, condense_nodes
import wio

__all__ = [
    'remove_nodes_outside_roi',
]

#seperated
def check_blobs_against_roi(experiment, x, y, r):
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
    return bids, are_inside

def remove_nodes_outside_roi(graph, experiment, x=None, y=None, r=None, ex_ids=None):
    """
    Removes nodes that are outside of a precalculated circle denoting a
    'region of interest'.  Must run before other simplifications; does not
    tolerate compound blob IDs

    params
    -----
    graph: (networkx graph)
       nodes are blob ids
    experiment: (multiworm Experiment)
       the experiment object corresponding to the same recording
    ex_id: (str)
       the experiment id (ie. timestamp) used to look up roi.
    """
    if isinstance(experiment, wio.Experiment):
        roi = experiment.prepdata.load('roi').set_index('bid')
        outside_nodes = roi[roi.inside_roi == False].index

    else:
        if not all([x, y, r]):
            raise ValueError('x, y, and r must be provided if the experiment '
                             'is not a wio.Experiment.')

        bids, are_inside = check_blobs_against_roi(experiment, x, y, r)
        outside_nodes = []
        for bid, in_roi in zip(bids, are_inside):
            if not in_roi:
                outside_nodes.append(bid)

    graph.remove_nodes_from(outside_nodes)
