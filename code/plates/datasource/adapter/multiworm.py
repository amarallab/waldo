# -*- coding: utf-8 -*-
"""
Gets data from the raw files specific blob ID
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import sys
import os.path as op

PROJECT_DIR = op.abspath(op.join(op.dirname(op.realpath(__file__)), '..', '..', '..', '..'))
#PROJECT_DIR = op.abspath(op.join(*((op.dirname(op.realpath(__file__)),) + ('..',)*4)))
MULTIWORM_DIR = op.join(PROJECT_DIR, 'code', 'shared', 'joining')
sys.path.append(MULTIWORM_DIR)

from multiworm.experiment import Experiment
import where

from .util import harmonize_id

def fetch_blob(*args):
    exp_id, blob_id = harmonize_id(*args)

    experiment = Experiment(where.where(exp_id))
    experiment.load_summary()

    blob = experiment.parse_blob(blob_id)

    return blob

def unified_blob(*args):
    """
    Adapts the "native" return value by fetch_blob() into what can be made
    identical to everything else.
    """
    blob = fetch_blob(*args)

    return {
        'centroid': blob['centroid'],
    }


