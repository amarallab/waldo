"""
Unified path management for files
"""
from __future__ import absolute_import

# standard library
import pathlib # py3.4+

# third party

# project specific
from . import directories

__all__ = [
    'calibration_data',
    'threshold_data',
    'threshold_cache',
]

def calibration_data(ex_id, root=None):
    directory = directories.prepdata(ex_id, root)
    filename = "{}-calibrationdata.json".format(ex_id)
    return directory / filename
    
def threshold_data(ex_id, root=None):
    directory = directories.prepdata(ex_id, root)
    filename = "{}-thresholddata.json".format(ex_id)
    return directory / filename

def threshold_cache(ex_id, root=None):
    directory = directories.prepdata(ex_id, root)
    filename = "threshold-cache.json"
    return directory / filename

def matches(ex_id):
    raise NotImplementedError()
    # used by waldo.annotation.image_validation.Validator (was broken though)
    directory = directories.prepdata(ex_id, root) / 'matches' # ???
    filename = '{eid}.csv'.format(eid=ex_id)
    return directory / filename # ???
