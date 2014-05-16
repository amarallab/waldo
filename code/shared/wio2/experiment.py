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
import contextlib

from .conf import settings
from .datafile import DataFile

FILE_EXT = 'h5'

class ExperimentAxisView(object):
    def __init__(self, experiment, key):
        self.experiment = experiment
        self.fixed_key = key


class ExperimentMeasurementView(ExperimentAxisView):
    def __getitem__(self, key):
        return self.experiment.read_measurement(key, self.fixed_key)

    def __iter__(self):
        # yields a tuple: (worm_id, DataFile)
        for result in self.experiment.read_measurements(self.fixed_key):
            yield result


class ExperimentWormView(ExperimentAxisView):
    def __getitem__(self, key):
        return self.experiment.read_measurement(self.fixed_key, key)

    def __iter__(self):
        # yields a tuple: (worm_id, DataFile)
        for result in self.experiment.read_worm(self.fixed_key):
            yield result


class Experiment(object):
    """
    Object to represent stored data from an experiment, real or otherwise.
    By default uses HDF5 to store time and data-series in a uniquely named
    folder, given by *experiment_id*, in either the given *data_root*
    directory or what is pulled from ``H5_DATA_ROOT`` in the configuration
    module.

    :class:`Experiment` can be opened to read or write data files, depending
    on *mode*.
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

    def __getitem__(self, key):
        if isinstance(key, six.string_types):
            return ExperimentMeasurementView(self, key)
        elif isinstance(key, six.integer_types):
            return ExperimentWormView(self, key)
        else:
            raise KeyError('Invalid key')

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
            worm_id = int(fn_parsed['worm_id'])
            series = fn_parsed['series']
            self.worms[worm_id][series] = fn
            self.measurements[series][worm_id] = fn

    def _filename(self, tag, worm_id=None, ext=None):
        """
        Returns a formatted filename using *tag*, often a measurement,
        *worm_id*, a unique "blob" identifier, and *ext*, the file extension.

        If *worm_id* is not provided, the filename will lack that field.
        *ext* defaults to the global ``FILE_EXT``
        """
        if ext is None:
            ext = FILE_EXT

        if worm_id is not None:
            fn = '{}_{:05d}-{}.{}'.format(
                    self.experiment_id, worm_id, tag, ext)
        else:
            fn = '{}-{}.{}'.format(
                    self.experiment_id, tag, ext)

        return fn

    def read_worms(self):
        """
        Generator that produces the worm ID and a dictionary with the
        measurement type as the key and payload as the value.  The type
        and particular format of the value can vary based on the storage
        type.
        """
        for worm_id, worm_measurements in six.iteritems(self.worms):
            # a context manager for all these files would be nice...
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
        for worm_id, filepath in six.iteritems(self.measurements[measurement]):
            with DataFile(str(filepath)) as datafile:
                yield worm_id, datafile

    def read_worm(self, worm_id):
        for measurement, filepath in six.iteritems(self.worms[worm_id]):
            with DataFile(str(filepath)) as datafile:
                yield measurement, datafile

    def read_measurement(self, worm_id, measurement):
        filepath = self.measurements[measurement][worm_id]
        with DataFile(str(filepath)) as datafile:
            return datafile.read_immediate()

    def write_measurement(self, worm_id, measurement, time, data, overwrite=False):
        """
        Write a *measurement* for a specific *worm_id*.

        Parameters
        ----------
        worm_id : int
            Worm identifier specific to experiment.
        measurement : str
            Name of the measurement.
        time : array-like
            Time points.
        data : array-like
            Data at each time point.  Must equal *time* in length.

        Keyword Arguments
        -----------------
        overwrite : bool
            Overwrite existing data without throwing an error.  Default False.
        """
        filepath = self.experiment_dir / self._filename(measurement, worm_id)
        with DataFile(str(filepath), 'w' if overwrite else 'w-') as datafile:
            datafile.write(time, data)

    @contextlib.contextmanager
    def open_auxillary(self, tag, ext=None, mode='r'):
        """
        Context manager to write an experiment file
        """
        filepath = self.experiment_dir / self._filename(tag, ext=ext)
        with filepath.open(mode) as f:
            yield f

    def write_auxillary(self, buf, *args, **kwargs):
        """
        Writes *buf* to an auxillary file.  See :func:`open_auxillary` for
        further call signature.
        """
        with self.open_auxillary(*args, **kwargs) as f:
            f.write(buf)
