"""
Unified path management tools
"""
from __future__ import absolute_import

# standard library
import os
import errno
import pathlib # py3.4+

# third party

# project specific

__all__ = [
    'mkdirp',
]

def mkdirp(path):
    """
    Recursivly creates path in filesystem, if it does not exist, a-la
    ``mkdir -p``
    """
    # http://stackoverflow.com/a/600612/194586

    # this should be reworked to use Path.mkdir...
    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
    path = str(path)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
