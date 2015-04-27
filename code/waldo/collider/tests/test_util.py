from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

def cumulative_sum(seq, start=0):
    x = start
    yield x
    for element in seq:
        x += element
        yield x
