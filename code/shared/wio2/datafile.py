# -*- coding: utf-8 -*-
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import h5py
import numpy as np

FIELDS = ['time', 'data']

class DataFile(h5py.File):
    """
    HDF5 file subclass used to read/write time and data series.
    """
    def __init__(self, *args, **kwargs):
        # default to read-only instead of R/W
        if len(args) < 2 and 'mode' not in kwargs:
            kwargs['mode'] = 'r'
        super(DataFile, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key not in FIELDS:
            raise KeyError()
        return super(DataFile, self).__getitem__(key)

    def __setitem__(self, key, data):
        if key not in FIELDS:
            raise KeyError()
        super(DataFile, self).__setitem__(key, data)

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
        Provide a lazy view to the file's datasets.
        """
        return [self[key] for key in ['time', 'data']]

    def read_immediate(self):
        """
        Read the datasets into memory immediately
        """
        time, data = self.read()
        return np.array(time), np.array(data)
