# -*- coding: utf-8 -*-
"""
Operations to get data out of simplified blob networks
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import pandas as pd

def blob_to_dataframe(experiment, blob_id, fields=None):
    """
    Loads the fields from a blob in the provided experiment and converts
    to a dataframe. 'frame' will always be tacked onto the list of fields,
    and only 'centroid' is included by default.
    """
    if fields is None:
        fields = ['centroid']
    fields = set(fields)
    fields.add('frame')

    blob = experiment[blob_id].crop(fields)

    df = pd.DataFrame(dict(blob))
    #df.set_index('frame', inplace=True)
    return df

def weighted_centroid(df):
    """
    Collapses a dataframe full of areas and centroids (presumably from the
    same frame) into a single area/centroid entry (dictionary)
    """
    n = len(df)
    area = sum(df['area']) / n
    centroid = [sum(coord) / n for coord in zip(*df['centroid'])]
    return {'area': area, 'centroid': centroid}

def polyblob_to_dataframe(experiment, graph, node):
    """
    Given an experiment and simplified graph, generate a unified dataframe
    with no duplicate frame rows for the specified node.
    """
    parts = components(graph, node)
    fields = ['area', 'centroid']
    blobs = [blob_to_dataframe(experiment, blob_id, fields=fields) for blob_id in parts]

    polyframe = pd.DataFrame()
    for blob in blobs:
        polyframe = polyframe.append(blob)

    frames = pd.Series(polyframe['frame'].values).unique()
    frames.sort()

    polyframe = {f: weighted_centroid(polyframe[polyframe.frame==f]) for f in frames}

    return pd.DataFrame.from_dict(polyframe, 'index')
