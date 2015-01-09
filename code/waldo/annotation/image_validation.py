'''
Filename: image_validation.py

Description: This class is used to get information about which
'''
from __future__ import (
        absolute_import, division, print_function, unicode_literals)

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard library
import sys
import os

# third party
import pandas as pd

# package specific
from waldo.wio import paths

class Validator(object):
    """ Class that is used to track data to validate the
    non-MWT image data.


    """
    def __init__(self, ex_id):
        """ ex_id """
        filename = paths.matches(ex_id)

        input_err_msg = '{eid} does not have validation file at: \
                         {p}'.format(eid=ex_id, p=str(filename))
        assert filename.is_file(), input_err_msg

        self.df = pd.read_csv(str(filename), index_col=0)
        self.df.fillna('', inplace=True)
        self.frames = sorted(list(set(self.df['frame'])))

    def show_frames(self):
        """ returns a sorted list with all frames that have been validated in this file. """
        return self.frames

    def full_check(self):
        """ returns a list of tuples specifying which blobs legitimate
            for all data regardless of frame.

        returns
        -----
        blob_check: (list of tuples)
            a list containing tuples in the following form: ( blob_id [int], is_good [bool])
        """
        tuples = [tuple(i) for i in self.df[['bid', 'good']].values]
        tuples = [(int(a), bool(b)) for (a,b) in tuples]
        return tuples

    def good_list(self):
        """ returns a list containing only good nodes.

        returns
        -----
        good_list: (list)
            a list containing blob_ids
        """
        tuples = [tuple(i) for i in self.df[['bid', 'good']].values]
        tuples = [(int(a), bool(b)) for (a,b) in tuples]
        good_list = [n for (n, g) in tuples if g]
        return good_list

    def bad_list(self):
        """ returns a list containing only bad nodes.

        returns
        -----
        bad_list: (list)
            a list containing blob_ids
        """
        tuples = [tuple(i) for i in self.df[['bid', 'good']].values]
        tuples = [(int(a), bool(b)) for (a,b) in tuples]
        bad_list = [n for (n, g) in tuples if not g]
        return bad_list


    def frame_check(self, frame):
        """ returns a list specifying which blobs in a given frame are legitimate.
            If frame is not present, return an empty list.

        params
        -----
        frame: (int)
            the frame you want to get information about

        returns
        -----
        blob_check: (list of tuples)
            a list containing tuples in the following form: ( blob_id [int], is_good [bool])
        """
        frame_set = self.df[self.df['frame'] == frame]
        tuples = [tuple(i) for i in frame_set[['bid', 'good']].values]
        tuples = [(int(a), bool(b)) for (a,b) in tuples]
        return tuples

    def joins(self):
        """ returns a list specifying all blobs that should be joined for the entire recording.

        returns
        -----
        blob_joins: (list of tuples)
            a list containing tuples in the following form: ( frame [int], 'blob1-blob2' [str])
        """
        joins = self.df[['frame', 'join']]
        joins = joins[joins['join'] != '']
        joins.drop_duplicates(cols='join', take_last=True, inplace=True)
        tuples = [tuple(i) for i in joins.values]
        tuples = [(int(a), [int(i) for i in b.split('-')]) for (a,b) in tuples]
        return tuples
