"""
Unified path management for directories
"""
from __future__ import absolute_import

# standard library
import pathlib # py3.4+

# third party

# project specific
from waldo.conf import settings

__all__ = [
    'prepdata',
    'experiment',
    'output',
]

def experiment(ex_id, root=None):
    """
    Input data
    """
    if root is None:
        root = settings.MWT_DATA_ROOT
    return pathlib.Path(root) / ex_id

def waldo_data(ex_id, root=None):
    if root is None:
        root = settings.PROJECT_DATA_ROOT
    return pathlib.Path(root) / ex_id

def prepdata(ex_id, root=None):
    """
    Where to store temporary processing data
    """
    return waldo_data(ex_id, root) / 'waldo'

def output(ex_id, root=None):
    """
    Where to put the cleaned blobs files
    """
    return waldo_data(ex_id, root) / 'blob_files'
