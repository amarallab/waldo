# -*- coding: utf-8 -*-
"""
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)


class ExperimentAnalyzer(object):
    def __init__(self):
        self.analyzers = []

    def add_analysis_method(self, analyzer):
        self.analyzers.append(analyzer)

    def analyze(self, blob_gen):
        for bid, blob in blob_gen:
            for analyzer in self.analyzers:
                analyzer.process_blob(blob)

    def results(self):
        data = {}
        for analyzer in self.analyzers:
            data.update(analyzer.result())

        return data

# import abc

# @six.add_metaclass(abc.ABCMeta)
class AnalysisMethod(object):
#     @abc.abstract_method
    def process_blob(self, blob):
        raise NotImplementedError()
