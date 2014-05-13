# -*- coding: utf-8 -*-
"""
WIO2 default configuration
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import pathlib

H5_DATA_ROOT = (pathlib.Path(__file__).parent
    / '..' # wio2
    / '..' # plates
    / '..' # code
    / '..' # (hg root)
    / 'data' / 'worms').resolve()

#STORAGE_TYPE = 'h5'
