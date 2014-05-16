# -*- coding: utf-8 -*-
"""
Assess the amount of noise present in an experiment
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import functools

import numpy as np

from .analytics import AnalysisMethod
from .smooth import smooth

SUBSAMPLE = 1

class SpeedEstimator(AnalysisMethod):
    """
    Attempt to determine the amount of noise present in some worm recordings.
    """
    def __init__(self, percentiles, smoothing):

        method, window = smoothing[:2]
        if len(smoothing) <= 2:
            params = ()
        else:
            params = smoothing[2:]

        #self.smoother = functools.partial(smooth, method, winlen=window, params=params)
        self.smoother = lambda series: smooth(method, series, window, *params)
        self.percentiles = percentiles

        self.speed_pctiles = []

    def process_blob(self, blob):
        xy = list(zip(*blob['xy_raw'][1]))
        xy_smoothed = [self.smoother(c) for c in xy]
        dxy = np.diff(np.array(xy_smoothed)[...,::SUBSAMPLE], axis=1)
        ds = np.linalg.norm(dxy, axis=0)

        self.speed_pctiles.append(np.percentile(ds, self.percentiles))

    def result(self):
        data = {
            'percentiles': self.speed_pctiles,
        }

        return {'speed': data}
