# -*- coding: utf-8 -*-
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import h5py

class DataFile(h5py.File):
    """
    HDF5 file subclass used to read/write time and data series.
    """
    def __init__(self, *args, **kwargs):
        # default to read-only instead of R/W
        if len(args) < 2 and 'mode' not in kwargs:
            kwargs['mode'] = 'r'
        super(DataFile, self).__init__(*args, **kwargs)

    def write(self, time, data):
        """
        Write a series of *data* points at associated *time* points.  The two
        series must be equal in length, but it doesn't matter if time is
        isochronous or not.
        """
        if len(time) != len(data):
            raise ValueError("time/data length mismatch")
        self['time'] = time
        self['data'] = data

    def read(self):
        """
        Load the data series from the file.
        """
        return (self[key] for key in ['time', 'data'])
