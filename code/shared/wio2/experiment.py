# -*- coding: utf-8 -*-
"""

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import re
import collections
import pathlib # stdlib 3.4+, pip <=3.3
import errno

from .conf import settings
from .datafile import DataFile

FILE_EXT = 'h5'

class Experiment(object):
    """

    """
    def __init__(self, experiment_id, mode='r', data_root=None):
        if data_root is None:
            data_root = settings.H5_DATA_ROOT

        if mode not in ['r', 'w']:
            raise ValueError('mode must be either "r"ead or "w"rite')

        self.experiment_id = experiment_id
        self.experiment_dir = pathlib.Path(data_root) / experiment_id

        self.worms = collections.defaultdict(dict)
        self.measurements = collections.defaultdict(dict)

        if mode == 'r':
            if not self.experiment_dir.exists():
                raise ValueError('Experiment data not found')

            self._index()

            # prevent key generation afterwards (throws KeyError on accessing
            # a non-existant element instead of silently making a new item)
            self.worms.default_factory = None
            self.measurements.default_factory = None

        elif mode == 'w':
            # initalize folder
            try:
                self.experiment_dir.mkdir(parents=True)
            except OSError as e:
                # ignore if the folder already existed.
                if e.errno != errno.EEXIST:
                    raise


    def _index(self):
        """
        Finds and adds all data files in the experiment directory to a
        pair of dictionaries that can be looped through in either
        direction.
        """
        fn_format = re.compile(self.experiment_id +
                r'_(?P<worm_id>\d{5})-(?P<series>\w+)')

        for fn in self.experiment_dir.glob('*.{}'.format(FILE_EXT)):
            fn_parsed = fn_format.match(fn.name)
            if fn_parsed is None:
                continue

            fn_parsed = fn_parsed.groupdict()
            worm_id = fn_parsed['worm_id']
            series = fn_parsed['series']
            self.worms[worm_id][series] = fn
            self.measurements[series][worm_id] = fn

    def _filename(self, worm_id, field):
        """UNTESTED
        """
        return '{exp_id}_{worm_id:05d}-{field}.{ext}'.format(
                exp_id=self.experiment_id, worm_id=worm_id, field=field,
                ext=FILE_EXT)

    def read_worms(self):
        """
        Generator that produces the worm ID and a dictionary with the
        measurement type as the key and payload as the value.  The type
        and particular format of the value can vary based on the storage
        type.
        """
        for worm_id, worm_measurements in six.iteritems(self.worms):
            worm_data = {
                    m: DataFile(str(filepath)) for (m, filepath)
                    in six.iteritems(worm_measurements)}
            yield worm_id, worm_data
            for datafile in six.itervalues(worm_data):
                datafile.close()

    def read_measurements(self, measurement):
        """
        Generator that produces the worm ID and *measurement* data loaded
        from the target file.  The type and particular format of the data
        can vary based on the storage type.
        """
        try:
            for worm_id, filepath in six.iteritems(self.measurements[measurement]):
                with DataFile(str(filepath)) as datafile:
                    yield worm_id, datafile
        except KeyError:
            return # the specified measurement has no measurements.

    def write_measurement(self, worm_id, measurement, time, data):
        """UNTESTED
        """
        filepath = self.experiment_dir / self._filename(worm_id, measurement)
        with DataFile(str(filepath), 'w') as datafile:
            datafile.dump(time, data)
