# -*- coding: utf-8 -*-
"""
FILL ME IN
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import abc

class DataNotFoundError(Exception):
    pass

@six.add_metaclass(abc.ABCMeta)
class WormDataAdapter(object):
    def __init__(self, experiment_id):
        self.exp_id = experiment_id

        self.locate()

    @abc.abstractmethod
    def locate(self):
        """Attempt to locate the data source"""
