# -*- coding: utf-8 -*-
"""
The only thing that understands Waldo, aside from Waldo.
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import os.path
import glob
import re
import collections

import pandas as pd

from .conf import settings
from .adapters import adapters

class WendasWorms(object):
    def __init__(self, experiment_id, search_remote=False, storage_type=None):
        if storage_type is None:
            storage_type = settings.STORAGE_TYPE
        self.experiment_id = experiment_id
        self.storage_adapter = adapters[storage_type]

        self.experiment_dir = os.path.join(settings.WORMS_DATA, self.experiment_id)
        if not os.path.isdir(self.experiment_dir):
            if not search_remote:
                raise ValueError('Experiment data not found')
            else:
                raise NotImplementedError()

        self.worms = collections.defaultdict(dict)
        self.measurements = collections.defaultdict(dict)

        fn_format = re.compile('(^|/)' + self.experiment_id + '_(?P<worm_id>\d{5})-(?P<series>\w+)')
        for fn in glob.iglob(os.path.join(self.experiment_dir, '*.{}'.format(self.storage_adapter.FILE_EXT))):
            fn_parsed = fn_format.search(fn)
            if fn_parsed is not None:
                fn_parsed = fn_parsed.groupdict()
                worm_id = fn_parsed['worm_id']
                series = fn_parsed['series']
                self.worms[worm_id][series] = fn
                self.measurements[series][worm_id] = fn

        self.worms.default_factory = None
        self.measurements.default_factory = None

    def available_measurements(self):
        """
        Returns a generator reflecting all measurement types saved into
        files.  Each measurement may not exist for every worm.
        """
        return six.iterkeys(self.measurements)

    def available_worms(self):
        """
        Returns a generator of all worm IDs.
        """
        return six.iterkeys(self.worms)

    def get_worms(self):
        """
        Generator that produces the worm ID and a dictionary with the
        measurement type as the key and payload as the value.  The type
        and particular format of the value can vary based on the storage
        type.
        """
        for worm_id, worm_measurements in six.iteritems(self.worms):
            yield worm_id, {m: self.storage_adapter.Worm(fn) for (m, fn) in six.iteritems(worm_measurements)}

    def get_measurements(self, measurement):
        """
        Generator that produces the worm ID and *measurement* data loaded
        from the target file.  The type and particular format of the data
        can vary based on the storage type.
        """
        for worm_id, datafile in six.iteritems(self.measurements[measurement]):
            yield worm_id, self.storage_adapter.Worm(datafile)
