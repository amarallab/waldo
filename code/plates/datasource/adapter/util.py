# -*- coding: utf-8 -*-
"""
General utility functions for data source functions
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six

def harmonize_id(*args):
    """
    Takes either a single value: "extended blob ID", or two: "experiment
    timestamp/ID" and "blob ID" and returns a string for the former of the
    latter and an integer for the latter's latter.
    """
    if len(args) == 1:
        exp_id, blob_id = args[0][:15], args[0][-5:]
    elif len(args) == 2:
        exp_id, blob_id = args
    else:
        raise ValueError("Invalid number of arguments")

    return exp_id, int(blob_id)
