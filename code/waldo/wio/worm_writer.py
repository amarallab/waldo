from __future__ import print_function, absolute_import, unicode_literals, division
"""
Preparation...pre-prepared, preprocessed information
"""
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library

# third party
import pandas as pd
import pathlib
# project specific
from . import paths
from waldo.conf import settings

class WormWriter(object):
    """
    Convienent interface to save and load "prep data" data frames.
    """
    def __init__(self, ex_id, save_root=None):
        self.id = ex_id
        if save_root:
            droot = settings.PROJECT_DATA_ROOT
            self.eid_directory = pathlib.Path(droot)
        else:
            d = pathlib.Path(settings.PROJECT_DATA_ROOT) / ex_id
            self.eid_directory = d

    def _filepath(self, bid, data_id):
        data_dir = self.eid_directory / data_id
        return data_dir / '{b}.csv'.format(b=bid)

    def load(self, bid, data_id, **kwargs):
        """ Load the specified *data_type* for a specific blob as a
        Pandas DataFrame. Keyword arguments are passed transparently
        to the pandas.read_csv function.
        """
        return pd.read_csv(str(self._filepath(bid, data_id)), **kwargs)

    def dump(self, bid, data_id, dataframe, **kwargs):
        """
        Dump the provided *dataframe* to a CSV file indicated by *data_type*.
        Keyword arguments are passed transparently to the DataFrame.to_csv
        method.
        """
        filepath = self._filepath(bid, data_id)
        filedir = self.eid_directory / data_id
        # ensure directory exists
        if not filedir.exists():
            filedir.mkdir()
        dataframe.to_csv(str(filepath), **kwargs)

    def data_ids(self):
        ignore_dirs = ['blob_files', 'waldo']
        data_id_dirs = []
        for d in self.eid_directory.glob('*'):
            if str(d.name) in ignore_dirs:
                continue
            if d.is_dir():
                data_id_dirs.append(d.name)
        return data_id_dirs

    def blobs(self, data_id):
        data_dir = self.eid_directory / data_id
        blobs = [b.stem for b in data_dir.glob('*.csv')]
        return blobs
