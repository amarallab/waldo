# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__all__ = [
    'csl',
]

def csl(l):
    return ', '.join(str(x) for x in l)
