# -*- coding: utf-8 -*-
"""
Load in blobs from whatever source and compare
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

from .adapter import hdf5
from .adapter import multiworm as mwa

ADAPTERS = {
    'raw': mwa.MultiwormAdapter,
    'h5': hdf5.WaldoAdapter,
}

def experiment(experiment_id, source):
    """
    Loads experiment data (maybe lazily).  Source can be one of:
      * ``raw``: Raw MWT data from *.blobs files and the like
      * ``h5``: Data pulled from Waldo-processed H5 files
    """

    if source not in ADAPTERS:
        raise ValueError("source must be one of {}".format(', '.join("'{}'".format(a) for a in ADAPTERS)))

    return ADAPTERS[source](experiment_id)

