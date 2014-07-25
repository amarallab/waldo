#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import random

# third party
import numpy as np
from scipy.stats import gaussian_kde

# project specific
from . import core

class Scorer(object):
    def __init__(self, experiment, move_threshold=core.REL_MOVE_THRESHOLD):
        self.experiment = experiment

        self._load_traces(move_threshold)

        self._generate_kdes()

    def _load_traces(self, move_threshold):
        good_ids = core.good_blobs(self.experiment, move_threshold)

        if core.TRACE_LIMIT_NUM and len(good_ids) > core.TRACE_LIMIT_NUM:
            good_ids = set(random.sample(good_ids, core.TRACE_LIMIT_NUM))

        self.traces = [
            core.absolute_displacement(self.experiment[x]['centroid'][:core.TRACE_LIMIT_FRAMES])
            for x in good_ids]

    def _generate_kdes(self, distance_samples=None, n_samples=100):
        traces = core.jagged_mask(self.traces)

        if distance_samples is None:
            max_dist = traces.compressed().max()
            distance_samples = np.linspace(0, max_dist, n_samples)
        else:
            n_samples = len(distance_samples)
            distance_samples = np.array(distance_samples)

        kde_fits = np.empty((traces.shape[1], n_samples))

        for n, distances in enumerate(traces[...,1:].T):
            kde_fits[n] = gaussian_kde(distances.compressed())(distance_samples)

        self.kde_fits = kde_fits

    def __call__(self, frame_gap, distance_gap):
        return 1/(frame_gap + distance_gap)
