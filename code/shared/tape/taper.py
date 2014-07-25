#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library

# third party

# project specific
from .scorer import Scorer

class Taper(object):
    """
    Designed to take a wio.Experiment-like object.
    """
    def __init__(self, experiment, regenerate_cache=False):
        self.experiment = experiment

        self.scorer = Scorer(experiment)

