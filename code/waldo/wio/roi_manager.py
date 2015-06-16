from __future__ import print_function, absolute_import, unicode_literals, division
import numpy as np

"""
Functions for creating masks for the region of interest
and for using the masks to check if points are within the region of interest.
"""

import six
from six.moves import (zip, filter, map, reduce, input, range)

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

# phase this out!
def check_points_against_roi(xs, ys, roi_dict):
    # dx = xs - roi['x']
    # dy = ys - roi['y']
    # img_roi_check = ((dx ** 2 + dy ** 2) <= roi['r'] ** 2)
    roi_mask = create_roi_mask(roi_dict)
    return are_points_inside_mask(xs, ys, roi_mask)


def are_points_inside_mask(xs, ys, mask):
    # this will break if points are outside mask
    # np arrays use y coordinates before x coordinates
    # TODO: fix this try and accept!
    try:
        return np.array([mask[yp, xp] for xp, yp in zip(xs, ys)])
    except:
        return np.array([mask[xp, yp] for xp, yp in zip(xs, ys)])

def create_roi_mask(d, shape=None):
    t = d.get('roi_type', 'circle')
    if shape is None:
        # shape = d.get('shape', (1728, 2352)) # backup if next line stops working again
        shape = d['shape']
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
    #print(min_y, max_y)
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