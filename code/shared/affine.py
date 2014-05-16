# -*- coding: utf-8 -*-
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np

def translate(x, y):
    return np.matrix([[1, 0, x],
                      [0, 1, y],
                      [0, 0, 1]])

def rotate(rad=None, deg=None):
    if rad is None:
        rad = np.radians(deg)

    return np.matrix([[+np.cos(rad), -np.sin(rad), 0],
                      [+np.sin(rad), +np.cos(rad), 0],
                      [      0     ,       0     , 1]])

def transform(coordinates, rotation=0, translation=(0, 0)):
    xy1 = np.ones((3, len(coordinates)))
    xy1[:2,...] = np.array(coordinates).T

    T = rotate(rotation) * translate(*translation)

    return (np.array((T * xy1)[:2]))
