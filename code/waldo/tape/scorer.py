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
import warnings

# third party
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import RectBivariateSpline
from scipy.signal import lfilter, boxcar

# project specific
from waldo.conf import settings
from . import core


class InsufficientData(Exception):
    pass


class MinimalData(Warning):
    pass


class Scorer(object):
    """
    Initialize with a wio.Experiment object.
    """
    def __init__(self, experiment, move_threshold=None):
        if move_threshold is None:
            move_threshold = settings.TAPE_REL_MOVE_THRESHOLD
        self.experiment = experiment

        self._interpolator = None
        self.max_speed = None

        self._load_traces(move_threshold)
        self._generate_kde_map()

        self.max_speed = self._find_max_speed()

    def _load_traces(self, move_threshold):
        good_ids = core.good_blobs(self.experiment, move_threshold)

        n_good_ids = len(good_ids)
        if n_good_ids == 0:
            print('move_threshold', move_threshold)
            print('in roi', self.experiment.in_roi())

            raise InsufficientData("No good traces identified.")
        elif n_good_ids <= settings.TAPE_MIN_TRACE_FAIL:
            raise InsufficientData(
                    "Only {} good trace(s) identified, we don't want to "
                    "generalize that to create scores.")
        elif n_good_ids <= settings.TAPE_MIN_TRACE_WARN:
            warnings.warn("A low number ({}) of good traces were identified; "
                          "take care with the results.".format(n_good_ids),
                          MinimalData)

        if (settings.TAPE_TRACE_LIMIT_NUM
                and n_good_ids > settings.TAPE_TRACE_LIMIT_NUM):
            state = random.getstate()
            random.seed(0) # pseudopseudorandom...
            # mostly for development, so any weirdness can be repeated
            good_ids = set(random.sample(good_ids, settings.TAPE_TRACE_LIMIT_NUM))
            random.setstate(state)

        self.traces = [
            core.absolute_displacement(self.experiment[x]['centroid']
                                       [:settings.TAPE_FRAME_SEARCH_LIMIT])
            for x in good_ids]

    def _find_max_speed(self):
        """Find maximum speed for later use"""
        window_size = (settings.TAPE_MAX_SPEED_SMOOTHING // 2) * 2 + 1 # round up to odd
        window = boxcar(window_size)
        window = window / sum(window)

        max_speed = 0
        for trace in self.traces:
            speed = np.diff(trace)
            smoothed = lfilter(window, 1, speed)[window_size // 2 + 1:]
            if len(smoothed):
                max_speed = max(max_speed, smoothed.max())

        return max_speed

    def _generate_kde_map(self, distance_samples=None, n_samples=None):
        if n_samples is None:
            n_samples = settings.TAPE_KDE_SAMPLES

        traces = core.jagged_mask(self.traces)[...,1:] # chop off first column (it's all zeroes)

        if distance_samples is None:
            max_dist = traces.compressed().max()
            distance_samples = np.linspace(0, max_dist, n_samples)
        else:
            n_samples = len(distance_samples)
            distance_samples = np.array(distance_samples)

        kde_fits = np.empty((traces.shape[1], n_samples))

        for n, distances in enumerate(traces.T):
            kde_fits[n] = gaussian_kde(distances.compressed())(distance_samples)

        self.framegap_domain = np.arange(1, kde_fits.shape[0] + 1)
        self.distance_domain = distance_samples
        self.kde_fits = kde_fits

        self._interpolator = RectBivariateSpline(
                x=self.framegap_domain,
                y=self.distance_domain,
                z=self.kde_fits)

    def __call__(self, frame_gap, distance_gap):
        if self._interpolator is None:
            return

        return float(max(self._interpolator(frame_gap, distance_gap), 1e-100))
