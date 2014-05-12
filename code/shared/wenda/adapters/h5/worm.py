# -*- coding: utf-8 -*-
"""

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import h5py

class Worm(h5py.File):
    def __init__(self, *args, **kwargs):
        # default to read-only instead of R/W
        if len(args) < 2 and 'mode' not in kwargs:
            kwargs['mode'] = 'r'
        super(Worm, self).__init__(*args, **kwargs)
