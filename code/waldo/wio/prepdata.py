from __future__ import print_function, absolute_import, unicode_literals, division
import numpy as np

"""
Preparation...pre-prepared, preprocessed information
"""
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library

# third party
import pandas as pd

# project specific
from . import paths


class PrepData(object):
    """
    Convienent interface to save and load "prep data" data frames.
    """
    def __init__(self, ex_id, prep_root=None):
        self.eid = ex_id
        if prep_root:
            self.directory = pathlib.Path(prep_root)
        else:
            self.directory = paths.prepdata(ex_id)

    def __getattr__(self, name):
        """
        For convienence
        prepdata.load('bounds') => prepdata.bounds
        """
        if name.startswith('_'):
            # pretend like we don't have this (because we shouldn't)
            # this fixes serialization problems in Python 3
            raise AttributeError()
        return self.load(name)

    def _filepath(self, data_type):
        return self.directory / '{}-{}.csv'.format(self.eid, data_type)

    def load(self, data_type, **kwargs):
        """
        Load the specified *data_type* as a Pandas DataFrame. Keyword
        arguments are passed transparently to the pandas.read_csv function.
        """
        return pd.read_csv(str(self._filepath(data_type)), **kwargs)

    def dump(self, data_type, dataframe, **kwargs):
        """
        Dump the provided *dataframe* to a CSV file indicated by *data_type*.
        Keyword arguments are passed transparently to the DataFrame.to_csv
        method.
        """
        # ensure directory exists
        if not self.directory.exists():
            self.directory.mkdir()

        dataframe.to_csv(str(self._filepath(data_type)), **kwargs)

    def good(self, frame=None):
        """ returns a list containing only good nodes.

        returns
        -----
        good_list: (list)
            a list containing blob_ids
        """
        if frame is None:
            df = self.load('matches')[['bid', 'good']]
        else:
            df = self.load('matches')
            df = df[df['frame'] == frame][['bid', 'good']]
        return [b for (b, v) in df.values if v]

    def bad(self, frame=None):
        """ returns a list containing only bad nodes.

        returns
        -----
        bad_list: (list)
            a list containing blob_ids
        """
        df = self.load('matches')[['bid', 'good']]

        if frame is None:
            df = self.load('matches')[['bid', 'good']]
        else:
            df = self.load('matches')
            df = df[df['frame'] == frame][['bid', 'good']]
        return [b for (b, v) in df.values if not v]


    def joins(self):
        """ returns a list specifying all blobs that should be joined
        according to the image data.

        returns
        -----
        blob_joins: (list of tuples)
            a list containing tuples in the following form: ( frame [int], 'blob1-blob2' [str])
        """
        joins = self.load('matches')[['frame', 'join']]
        joins = joins[joins['join'] != '']
        joins.drop_duplicates(cols='join', take_last=True, inplace=True)
        tuples = [tuple(i) for i in joins.values]
        tuples = [(int(a), [int(i) for i in b.split('-')]) for (a,b) in tuples]
        return tuples

    def outside(self, frame=None):
        df = self.load('roi')[['bid', 'inside_roi']]
        return [b for (b, v) in df.values if not v]

    def moved(self, bl_threhold=2):
        pass