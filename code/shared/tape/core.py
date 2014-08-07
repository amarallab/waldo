#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import random

# third party
import numpy as np

# project specific
#import..

def absolute_displacement(centroid, reverse=False):
    """
    Given a sequence of X-Y coordinates, return the absolute distance from
    each point to the first point.

    Parameters
    ----------
    centroid : array_like
        A sequence of X-Y coordinate pairs.

    Keyword Arguments
    -----------------
    reverse : bool
        Reverse the centroid sequence before taking the absolute values
    """
    if reverse:
        centroid = centroid[::-1]
    centroid = np.array(centroid)
    centroid -= centroid[0]

    # http://stackoverflow.com/a/12712725/194586
    displacement = np.sqrt(np.einsum('...i,...i', centroid, centroid, dtype='float64'))
    return displacement

def good_blobs(experiment, move_threshold):
    return experiment.in_roi() & experiment.rel_move(move_threshold)

def jagged_mask(data):
    """
    Stacks up variable length series into a faux-jagged (masked) array.
    """
    h_size = max(len(trace) for trace in data)
    # stack up data, mask off non-existant values (np can't do jagged arrays)
    dstack = np.ma.masked_all((len(data), h_size))
    for i, trace in enumerate(data):
        len_trace = len(trace)
        dstack[i, 0:len_trace] = trace

    return dstack
