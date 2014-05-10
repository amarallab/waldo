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
        return list(six.iterkeys(self.measurements))

    def available_worms(self):
        return list(six.iterkeys(self.worms))

    def get_measurements(self, measurement):
        for worm_id, datafile in six.iteritems(self.measurements[measurement]):
            yield worm_id, self.storage_adapter.Worm(datafile)
