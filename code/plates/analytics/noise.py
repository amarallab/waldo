# -*- coding: utf-8 -*-
"""
Assess the amount of noise present in an experiment
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np
import scipy.optimize as spo

from .analytics import AnalysisMethod

def normpdf(x, *args):
    "Return the normal pdf evaluated at *x*; args provides *mu*, *sigma*"
    # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mlab.py#L1554
    mu, sigma = args
    return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5 * (1./sigma*(x - mu))**2)

def fit_gaussian(x, num_bins=200):
    # some testdata has no variance whatsoever, this is escape clause
    if abs(max(x) - min(x)) < 1e-5:
        print('fit_gaussian exit')
        return max(x), 0

    n, bin_edges = np.histogram(x, num_bins, normed=True)
    bincenters = [0.5 * (bin_edges[i + 1] + bin_edges[i]) for i in range(len(n))]

    # Target function
    fitfunc = lambda p, x: normpdf(x, p[0], p[1])

    # Distance to the target function
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    # Initial guess for the parameters
    mu = np.mean(x)
    sigma = np.std(x)
    p0 = [mu, sigma]
    p1, success = spo.leastsq(errfunc, p0[:], args=(bincenters, n))
    # weirdly if success is an integer from 1 to 4, it worked.
    if success in [1,2,3,4]:
        mu, sigma = p1
        return mu, sigma
    else:
        return None

def centroid_steps(centroid):
    xy = zip(*centroid)
    dxy = [np.diff(d) for d in xy]
    return dxy

def centroid_stats(steps):
    stats = []
    for data in steps:
        # div by root 2 assumes pure gaussian noise from a stationary mean
        stats.append(fit_gaussian(data)/np.sqrt(2))
    return stats


class NoiseEstimator(AnalysisMethod):
    """
    Attempt to determine the amount of noise present in some worm recordings.
    """
    def __init__(self):
        self.std_devs = []
        self.means = []

    def process_blob(self, blob):
        """
        Feed parsed blobs and it generates the appropriate statistics.
        """
        if blob is None:
            return

        steps = centroid_steps(blob['xy_raw']['data'])
        result = centroid_stats(steps)
        means, sds = zip(*result)

        self.std_devs.append(sds)
        self.means.append(means)

    def result(self):

        mean_mean = np.mean(self.means, axis=0)
        mean_std_dev = np.mean(self.std_devs, axis=0)

        data = {
            'mean_xy': mean_mean.tolist(),
            'std_dev_xy': mean_std_dev.tolist(),
            'means': self.means,
            'std_devs': self.std_devs,
        }

        return {'noise': data}
