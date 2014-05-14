# -*- coding: utf-8 -*-
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np

import wio2

class SyntheticWorm(object):
    def __init__(self, n_worms=10, n_points=3600, **params):
        self.n_worms = n_worms
        self.n_points = n_points

        self.params = self.REALIZATION_DEFAULTS.copy()
        self.params.update(params)

        self.worms_true = np.empty((self.n_worms, self.n_points, 2))
        self.worms = None

    def corrupt(self, noise_sd):
        worms = np.copy(self.worms_true)
        if noise_sd == 0:
            self.worms = worms
        else:
            noise = np.random.normal(scale=noise_sd, size=self.worms.shape)
            self.worms = worms + noise

    def realize_all(self):
        self.worms_true = np.empty((self.n_worms, self.n_points, 2))
        for worm in self.worms_true:
            np.copyto(worm, self.realization())

        # NOTE: ref sharing; don't modify in-place or "true" values will also
        # change.
        self.worms = self.worms_true

    def dump(self, experiment_id):
        experiment = wio2.Experiment(experiment_id, mode='w')
        for worm_id, worm in enumerate(self.worms):
            print('writing worm id {}, size: {}'.format(worm_id, worm.shape))
            experiment.write_measurement(worm_id, 'xy_raw', *worm.T)
