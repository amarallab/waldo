# -*- coding: utf-8 -*-
"""
Wenda default configuration
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

from os.path import join, dirname

WORMS_DATA = join(dirname(__file__),
    '..', # wanda
    '..', # plates
    '..', # code
    '..', # (hg root)
    'data', 'worms')

STORAGE_TYPE = 'h5'
