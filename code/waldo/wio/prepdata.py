from __future__ import print_function, absolute_import, unicode_literals, division
import numpy as np

"""
Preparation...pre-prepared, preprocessed information
"""
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library

# third party
import pandas as pd

# project specific
from . import paths

class PrepData(object):
    """
    Convienent interface to save and load "prep data" data frames.
    """
    def __init__(self, ex_id, prep_root=None):
        self.eid = ex_id
        if prep_root:
            self.directory = pathlib.Path(prep_root)
        else:
            self.directory = paths.prepdata(ex_id)

    def __getattr__(self, name):
        """
        For convienence
        prepdata.load('bounds') => prepdata.bounds
        """
        if name.startswith('_'):
            # pretend like we don't have this (because we shouldn't)
            # this fixes serialization problems in Python 3
            raise AttributeError()
        return self.load(name)

    def _filepath(self, data_type):
        return self.directory / '{}-{}.csv'.format(self.eid, data_type)

    def load(self, data_type, **kwargs):
        """
        Load the specified *data_type* as a Pandas DataFrame. Keyword
        arguments are passed transparently to the pandas.read_csv function.
        """
        return pd.read_csv(str(self._filepath(data_type)), **kwargs)

    def dump(self, data_type, dataframe, **kwargs):
        """
        Dump the provided *dataframe* to a CSV file indicated by *data_type*.
        Keyword arguments are passed transparently to the DataFrame.to_csv
        method.
        """
        # ensure directory exists
        if not self.directory.exists():
            self.directory.mkdir()

        dataframe.to_csv(str(self._filepath(data_type)), **kwargs)

    def good(self, frame=None):
        """ returns a list containing only good nodes.

        returns
        -----
        good_list: (list)
            a list containing blob_ids
        """
        if frame is None:
            df = self.load('matches')[['bid', 'good']]
        else:
            df = self.load('matches')
            df = df[df['frame'] == frame][['bid', 'good']]
        return [b for (b, v) in df.values if v]

    def bad(self, frame=None):
        """ returns a list containing only bad nodes.

        returns
        -----
        bad_list: (list)
            a list containing blob_ids
        """
        df = self.load('matches')[['bid', 'good']]

        if frame is None:
            df = self.load('matches')[['bid', 'good']]
        else:
            df = self.load('matches')
            df = df[df['frame'] == frame][['bid', 'good']]
        return [b for (b, v) in df.values if not v]


    def joins(self):
        """ returns a list specifying all blobs that should be joined
        according to the image data.

        returns
        -----
        blob_joins: (list of tuples)
            a list containing tuples in the following form: ( frame [int], 'blob1-blob2' [str])
        """
        joins = self.load('matches')[['frame', 'join']]
        joins = joins[joins['join'] != '']
        joins.drop_duplicates(cols='join', take_last=True, inplace=True)
        tuples = [tuple(i) for i in joins.values]
        tuples = [(int(a), [int(i) for i in b.split('-')]) for (a,b) in tuples]
        return tuples

    def outside(self, frame=None):
        df = self.load('roi')[['bid', 'inside_roi']]
        return [b for (b, v) in df.values if not v]

    def moved(self, bl_threhold=2):
        pass


def roi_dict_to_points(d):
    t = d.get('roi_type', 'circle')
    if t == 'circle':
        # draw full circle region of interest
        roi_t = np.linspace(0, 7, 100)
        roi_x = d['r'] * np.cos(roi_t) + d['x']
        roi_y = d['r'] * np.sin(roi_t) + d['y']
        return roi_x, roi_y
    elif t == 'polygon':
        roi_x, roi_y = zip(*d['points'])
        return roi_x, roi_y
    else:
        #TODO: error
        return None


# def check_points_against_roi(xs, ys, roi_dict):
#     dx = xs - roi['x']
#     dy = ys - roi['y']
#     img_roi_check = ((dx ** 2 + dy ** 2) <= roi['r'] ** 2)


def are_points_inside_mask(xs, ys, mask):
    # this will break if points are outside mask
    # np arrays use y coordinates before x coordinates
    return np.array([mask[yp, xp] for xp, yp in zip(xs, ys)])


def create_roi_mask(d, shape=None):
    t = d.get('roi_type', 'circle')
    if shape is None:
        shape = d['shape']
    print('using shape:', shape)
    if t == 'circle':
        return create_roi_mask_circle(d['x'], d['y'], d['r'], shape)
    elif t == 'polygon':
        return create_roi_mask_polygon(d['points'], shape)
    else:
        #TODO: error
        return None


def create_roi_mask_circle(x, y, r, shape):
    ny, nx = shape
    xs = np.arange(0, nx)
    ys = np.arange(0, ny)
    xv, yv = np.meshgrid(xs, ys)
    dy = yv - y
    dx = xv - x
    d = np.sqrt(dy ** 2 + dx ** 2) #.T no longer using transpose
    roi_mask = d <= r
    return roi_mask


def create_roi_mask_polygon(points, shape):
    roi_mask = np.zeros(shape=shape, dtype=bool)
    _fill_polygon(roi_mask, points)
    return roi_mask


def _pairwise(it):
    ''' utilitiy function for creating polygon'''
    it = iter(it)
    while True:
        yield next(it), next(it)


def _fill_polygon(img, points):
    ''' utilitiy function for creating polygon'''
    yy = [y for x, y in points]
    min_y = int(min(yy))
    max_y = int(max(yy)) + 1 # hacky solution to rounding up
    print(min_y, max_y)
    for y in range(min_y, max_y):
        cut_points = []
        p0 = points[-1]
        for p1 in points:
            sign = p0[1] - p1[1]
            if sign != 0:
                (x0, y0), (x1, y1) = zip(*sorted([p0, p1]))
                if sign < 0:
                    x0, y0 = p0
                    x1, y1 = p1
                else:
                    x0, y0 = p1
                    x1, y1 = p0

                if y0 <= y < y1:
                    m = float(x1 - x0) / (y1 - y0)
                    cx = int(x0 + (y - y0) * m)
                    cut_points.append(cx)
            p0 = p1
        cut_points = sorted(cut_points)
        for x0, x1 in _pairwise(cut_points):
            for x in range(x0, x1):
                #img[x, y] = True
                img[y, x] = True # turns out that in array operations y is before x