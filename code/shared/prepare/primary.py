from __future__ import print_function, absolute_import, unicode_literals, division
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


class DataFrameBuilder(object):
    def __init__(self):
        self.data = []

    def render(self):
        return pd.DataFrame(self.data)


class BoundsBuilder(DataFrameBuilder):
    name = 'bounds'

    def append(self, bid, blob):
        if not blob['centroid']:
            return

        centroid_x, centroid_y = zip(*blob['centroid'])
        x_min, x_max = min(centroid_x), max(centroid_x)
        y_min, y_max = min(centroid_y), max(centroid_y)

        self.data.append({'bid': bid,
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
            })


class SizeBuilder(DataFrameBuilder):
    name = 'sizes'

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
                    'bid':bid,
                    'area_median':area,
                    'midline_median':midline_median
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
    name = 'terminals'

    def append(self, bid, blob):
        if not blob['centroid']:
            return

        times = blob['time']
        frames = blob['frame']

        t0, tN = min(times), max(times)
        f0, fN = min(frames), max(frames)
        x0, y0 = blob['centroid'][0]
        xN, yN = blob['centroid'][-1]

        self.data.append({'bid': bid,
                'x0':x0, 'xN':xN,
                'y0':y0, 'yN':yN,
                't0':t0, 'tN':tN,
                'f0':f0, 'fN':fN,
            })


def summarize(experiment):
    """
    Given an experiment, generate 3 data frames returned as values in a
    dictionary.

    1. Bounds: in x and y, the max and min position each blob moved.
    2. Terminals: the position (x and y) at blob end and begin, as well
        as the time in seconds and frames.
    3. Sizes: median sizes of the area and midline provided by the raw
        blob data.
    """
    #bounds_data, terminals_data, sizes_data = [], [], []
    bounds = BoundsBuilder()
    terminals = TerminalsBuilder()
    sizes = SizeBuilder()
    builders = [bounds, terminals, sizes]

    for bid, blob in experiment.blobs():
        try:
            for builder in builders:
                builder.append(bid, blob)
            # if blob['centroid']:
                # times = blob['time']
                # frames = blob['frame']
                # centroid_x, centroid_y = zip(*blob['centroid'])
                # x_min, x_max = min(centroid_x), max(centroid_x)
                # y_min, y_max = min(centroid_y), max(centroid_y)
                # t0, tN = min(times), max(times)
                # f0, fN = min(frames), max(frames)
                # bounds_data.append({'bid': bid, 'x_min':x_min,
                #                     'x_max': x_max, 'y_min':y_min,
                #                     'y_max': y_max})

                # x0, y0 = blob['centroid'][0]
                # xN, yN = blob['centroid'][-1]
                # terminals_data.append({'bid': bid, 'x0':x0, 'xN':xN,
                #                        'y0':y0, 'yN':yN,
                #                        't0':t0, 'tN':tN,
                #                        'f0':f0, 'fN':fN})

            # midline_median, area = np.nan, np.nan
            # if blob['midline'] and any(blob['midline']):
            #     midline_median = np.median([_midline_length(p) for p in blob['midline'] if p])
            # if blob['area']:
            #     area = np.median(blob['area'])
            # if blob['midline'] or blob['area']:
            #     sizes_data.append({'bid':bid, 'area_median':area, 'midline_median':midline_median})

        except KeyError:
            # zero frame blobs
            assert blob.blob_data == {}
            pass

    return {'bounds': bounds.render(), 'terminals': terminals.render(), 'sizes': sizes.render()}
