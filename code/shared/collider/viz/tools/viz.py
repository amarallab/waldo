# -*- coding: utf-8 -*-
"""
MWT collision graph visualizations - Before and after outlines
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np
from matplotlib.patches import Polygon, Wedge
from scipy import ndimage

from .box import Box

__all__ = [
    'patch_contours',
    'get_image',
    'get_contour',
]

def patch_contours(contours):
    """
    Convert all the contours (Nx2 series of points) into matplotlib patches
    and return the enclosing bounding box and a list of patches.
    """
    bounds = []
    patches = []
    for contour in contours:
        contour = np.array(contour)
        if contour.shape == (2,):
            contour = np.reshape(contour, (-1, 2))
        bbox = Box.fit(contour)
        if len(contour) == 1:
            patches.append(Wedge(contour, 10, 0, 360, 5, alpha=0.6))
            bbox.size = 30, 30
        else:
            patches.append(Polygon(contour, closed=True, alpha=0.3))
        bounds.append(bbox)

    bounds = sum(bounds)
    return bounds, patches

def get_image(experiment, **kwargs):
    """
    Find the nearest image (pass 'time' or 'frame' as keyword argument) in
    the experiment and load into a Numpy array.
    """
    im_file, actual_time = experiment.image_files.nearest(**kwargs)
    img = ndimage.imread(str(im_file))
    return img

def get_contour(blob, index='first', centroid_fallback=True):
    if index not in ['first', 'last']:
        raise ValueError('index must be either "first" or "last"')

    method = index + '_valid_index'

    col = 'contour'
    index = getattr(blob.df[col], method)()
    if index is None:
        if not centroid_fallback:
            raise KeyError('No valid index')
        col = 'centroid'
        index = getattr(blob.df[col], method)()

    return blob.df[col][index]
