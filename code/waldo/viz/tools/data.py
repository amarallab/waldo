# -*- coding: utf-8 -*-
"""
Load, parse, process data
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools
import collections

from waldo.extern import multiworm
from multiworm.blob import BlobDataFrame
from multiworm.readers.blob import parse as parse_blob

__all__ = [
    'frame_dataframe',
    'fill_empty_contours',
]

def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    # https://docs.python.org/3/library/itertools.html#itertools-recipes
    return next(itertools.islice(iterable, n, None), default)

def frame_raw_line(blob, frame):
    "Grab the raw line for the given blob on frame"
    return nth(blob.raw_lines(), frame - blob.born_f)

def frame_data(experiment, frame):
    "Grab all the data on a given frame and parse"
    data = []
    for bid in experiment.blobs_in_frame(frame):
        try:
            datum = parse_blob([frame_raw_line(experiment[bid], frame)])
            datum['bid'] = [bid]
            # everything's a list of 1 thing
            data.append({k: v[0] for k, v in six.iteritems(datum)})
        except ValueError:
            # probably trying to access an empty blob
            pass
    return data

def frame_dataframe(experiment, frame):
    "Convert frame data into a dataframe (ha) and decode contours"
    data = frame_data(experiment, frame)
    data = BlobDataFrame(data)
    data.decode_contour()
    assert all(data['frame'] == frame)
    data.drop('frame', axis=1)
    return data

def fill_empty_contours(df):
    def _fill(series):
        if series['contour']:
            return series['contour']
        return series['centroid']

    df['contour'] = df.apply(_fill, axis=1)
    return df
