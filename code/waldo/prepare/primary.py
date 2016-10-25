from __future__ import print_function, absolute_import, division
"""
Generate summary data directly from the raw experiment. All should happen
at once to avoid having to re-parse or otherwise hold massive amounts of
data in memory at the same time.
"""
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library

# third party
import numpy as np
import pandas as pd

# project specific

__all__ = ['summarize']

class DataFrameBuilder(object):
    column_order = None
    def __init__(self):
        self.data = []

    def render(self):
        return pd.DataFrame(self.data, columns=self.column_order)


class BoundsBuilder(DataFrameBuilder):
    data_type = 'bounds'
    column_order = ['bid', 'x_min', 'x_max', 'y_min', 'y_max']

    def append(self, bid, blob):
        if not blob['centroid']:
            return

        centroid_x, centroid_y = zip(*blob['centroid'])
        self.data.append({'bid': bid,
                'x_min': min(centroid_x), 'x_max': max(centroid_x),
                'y_min': min(centroid_y), 'y_max': max(centroid_y),
            })


class SizeBuilder(DataFrameBuilder):
    data_type = 'sizes'
    column_order = ['bid', 'area_median', 'midline_median']

    def append(self, bid, blob):
        no_data = True

        if blob['midline'] and any(blob['midline']):
            midline_median = np.median([self._midline_length(p) for p in blob['midline'] if p])
            no_data = False
        else:
            midline_median = np.nan

        if blob['area']:
            area = np.median(blob['area'])
            no_data = False
        else:
            midline_median = np.nan

        if not no_data:
            self.data.append({
                    'bid': bid,
                    'area_median': area,
                    'midline_median': midline_median
                })

    def _midline_length(self, points):
        """
        Calculates the length of a path connecting *points*.
        """
        x, y = zip(*points)
        dx = np.diff(np.array(x))
        dy = np.diff(np.array(y))
        return np.sqrt(dx**2 + dy**2).sum()


class TerminalsBuilder(DataFrameBuilder):
    data_type = 'terminals'
    column_order = ['bid', 'x0', 'y0', 't0', 'f0',
                           'xN', 'yN', 'tN', 'fN']

    def append(self, bid, blob):
        if not blob['centroid']:
            return

        x0, y0 = blob['centroid'][0]
        xN, yN = blob['centroid'][-1]

        self.data.append({'bid': bid,
                'x0': x0, 'xN': xN,
                'y0': y0, 'yN': yN,
                't0': min(blob['time']),  'tN': max(blob['time']),
                'f0': min(blob['frame']), 'fN': max(blob['frame']),
            })

def create_primary_df(experiment, df_type=None, callback=None):
    builders = {'bounds': BoundsBuilder(),
                'terminals': TerminalsBuilder(),
                'sizes': SizeBuilder()}

    assert df_type in builders, '{t} not a real type of df'.format(t=df_type)
    builder = builders[df_type]
    count = 0
    for _ in experiment.blobs():
        count += 1
    blobs = experiment.blobs()
    for i, (bid, blob) in enumerate(blobs):
        try:
            builder.append(bid, blob)
        except KeyError:
            # zero frame blobs
            pass
        if callback:
            callback(i/count)
    if callback:
        callback(1)
    return builder.render()

#TODO this fails on big recordings due to memory errors
def summarize(experiment, callback=None):
    """
    Given an experiment, generate 3 data frames returned as values in a
    dictionary.

    1. Bounds: in x and y, the max and min position each blob moved.
    2. Terminals: the position (x and y) at blob end and begin, as well
        as the time in seconds and frames.
    3. Sizes: median sizes of the area and midline provided by the raw
        blob data.

    The callback (if not ``None``) is called with fractional progression (0
    to 1)
    """
    builders = [BoundsBuilder(), TerminalsBuilder(), SizeBuilder()]

    for i, (bid, blob) in enumerate(experiment.blobs()):
        try:
            for builder in builders:
                builder.append(bid, blob)
        except KeyError:
            # zero frame blobs
            pass

        if callback:
            callback(i / len(experiment))
    print('done building')
    if callback:
        callback(1)

    return {builder.data_type: builder.render() for builder in builders}
