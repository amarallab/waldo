# -*- coding: utf-8 -*-
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import abc
import json

import numpy as np

import wio2

@six.add_metaclass(abc.ABCMeta)
class SyntheticWorm(object):
    """
    Base class for generation of synthetic data
    """

    MODEL_DEFAULTS = {}

    def __init__(self, n_worms=10, n_points=3600, dt=1, **params):
        self.n_worms = n_worms
        self.n_points = n_points
        self.times = np.arange(n_points) * dt

        self.params = self.MODEL_DEFAULTS.copy()
        self.params.update(params)

        self.worms_true = np.empty((self.n_worms, self.n_points, 2))
        self.worms = None

    def __iter__(self):
        # make this look sorta like a wio2.Experiment
        for worm_id, worm in enumerate(self.worms, start=1):
            #yield worm_id, {'xy_raw': {'time': self.times, 'data': worm}}
            yield worm_id, {'xy_raw': (self.times, worm)}

    def corrupt(self, noise_sd):
        worms = np.copy(self.worms_true)
        if noise_sd == 0:
            self.worms = worms
        else:
            noise = np.random.normal(scale=noise_sd, size=self.worms.shape)
            self.worms = worms + noise

    @abc.abstractmethod
    def realization(self):
        return np.zeros((self.n_points, 2))

    def realize_all(self):
        self.worms_true = np.empty((self.n_worms, self.n_points, 2))
        for worm in self.worms_true:
            np.copyto(worm, self.realization())

        # NOTE: ref sharing; don't modify in-place or "true" values will also
        # change.
        self.worms = self.worms_true

    def write(self, experiment_id):
        experiment = wio2.Experiment(experiment_id, mode='w')
        for worm_id, worm in enumerate(self.worms):
            experiment.write_measurement(worm_id, 'xy_raw', self.times, worm, overwrite=True)

        with experiment.open_auxillary('model_parameters', 'json', mode='w') as f:
            buf = json.dumps(self.params, indent=4)
            f.write(six.text_type(buf))
