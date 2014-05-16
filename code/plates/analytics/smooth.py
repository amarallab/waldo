# -*- coding: utf-8 -*-
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import zip, range, map

import numpy as np
import scipy.signal as ss

from .sgolay import savitzky_golay

# Filters that are included in scipy.signal
BASE_METHODS = [
    'boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett',
    'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann',
    'kaiser', 'gaussian', 'general_gaussian', 'slepian', 'chebwin'
]

def sgolay(series, window, order):
    series = np.array(series)
    window = int(window)
    order = int(order)
    return savitzky_golay(series, window, order)

ADDITIONAL_METHODS = {
    'sgolay': sgolay,
}
METHODS = BASE_METHODS + list(six.iterkeys(ADDITIONAL_METHODS))

def smooth(method, series, winlen, *params):
    if method in ADDITIONAL_METHODS:
        return ADDITIONAL_METHODS[method](series, winlen, *params)

    winlen = int(winlen) // 2 * 2 + 1 # make it odd, rounding up
    half_win = winlen // 2 # ignoring the 0.5 remainder
    wintype = (method,) + tuple(int(x) for x in params)
    try:
        fir_win = ss.get_window(wintype, winlen)
    except ValueError:
        raise ValueError('Unrecognized smoothing type')

    b = fir_win / sum(fir_win)
    a = [1]
    #zi = ss.lfiltic(b, a)
    #zi = series[0] * np.ones(len(b) - 1)
    return ss.lfilter(b, a, series)[winlen-1:]
