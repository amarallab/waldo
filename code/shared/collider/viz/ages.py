# -*- coding: utf-8 -*-
"""
MWT collision graph visualizations - Ages of blobs
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from ..simplifications.util import lifespan

def age_distribution(digraph):
    ages = [lifespan(digraph, node) for node in digraph]
    age_ccdf = cdf(ages, ccdf=True)

    fig, ax = plt.subplots()
    ax.semilogx(*age_ccdf)
    ax.set_title("Blob node lifespan")
    ax.set_xlabel("Life (frames)")
    ax.set_ylabel("CDF")
    ax.text(0.95, 0.05, 'n = {}'.format(len(ages)),
            ha='right', va='baseline', transform=ax.transAxes)

    return fig, ax

def cdf(dist, ccdf=False, norm=True, stepped=False):
    '''
    Generate a cumulative distribution function (CDF) of *dist*.  *dist* must
    be an iterable.  Returns (x, CDF(x)).

    Keyword Parameters
    ------------------
    ccdf : bool
        If True, the returned data is a complementary CDF, beginning at 1
        and decaying to 0.
    norm : bool
        If True, the range of the CDF is normalized to [0, 1]
    stepped : bool
        If True, the returned values show discontinuties at each value.  As
        a result, the length of the returned arrays are 2*length(dist)
    '''
    dist = np.array(dist)
    dist.sort()

    L = len(dist)
    step = 1/L if norm else 1
    cdf_ = np.arange(0, L*step, step)

    if stepped:
        dist = dist.repeat(2)
        cdf_ = np.vstack((cdf_, cdf_+step)).ravel(order='F')

    if ccdf:
        cdf_ = L*step - cdf_

    return dist, cdf_
