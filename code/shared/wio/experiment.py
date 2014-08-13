#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A specialized Waldo version of the Experiment class that contains specific
features.
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np
import pandas as pd

import multiworm
from . import file_manager as fm

class Experiment(multiworm.Experiment):
    """
    Augment multiworm's Experiment with the auxillary PrepData class
    available as the ``prepdata`` attribute.
    """
    def __init__(self, *args, **kwargs):
        super(Experiment, self).__init__(*args, **kwargs)
        self.prepdata = fm.PrepData(self.experiment_id)

        self._prep_df = None
        self._typical_bodylength = None

    def _pull_prepdata(self):
        bounds = self.prepdata.load('bounds')
        sizes = self.prepdata.load('sizes')

        self._prep_df = pd.merge(bounds, sizes, on='bid')

    def in_roi(self):
        if self._prep_df is None:
            self._pull_prepdata()

        if 'in_roi' not in self._prep_df.columns:
            prep_file = fm.ImageMarkings(ex_id=self.experiment_id)
            roi = prep_file.roi()

            x_mid = (self._prep_df.x_min + self._prep_df.x_max) / 2
            y_mid = (self._prep_df.y_min + self._prep_df.y_max) / 2

            self._prep_df['in_roi'] = (x_mid - roi['x']) ** 2 + (y_mid - roi['y']) ** 2 < roi['r'] ** 2

        in_roi = set(
                bid
                for bid, is_in
                in zip(self._prep_df.bid, self._prep_df.in_roi)
                if is_in)

        return in_roi

    def rel_move(self, threshold):
        if self._prep_df is None:
            self._pull_prepdata()

        if 'rel_move' not in self._prep_df.columns:
            movement_px = (self._prep_df.x_max - self._prep_df.x_min) + (self._prep_df.y_max - self._prep_df.y_min)

            self._prep_df['rel_move'] = movement_px / self._prep_df.midline_median

        moved_enough = set(
                bid
                for bid, moved
                in zip(self._prep_df.bid, self._prep_df.rel_move)
                if moved >= threshold)

        return moved_enough

    @property
    def typical_bodylength(self):
        if self._typical_bodylength is None:
            # find out the typical body length if we haven't already
            im_df = self.prepdata.load('matches')
            matched_blobs = im_df[im_df['good'] & im_df['roi']]['bid']

            sizes = self.prepdata.load('sizes')
            sizes.set_index('bid', inplace=True)

            good_midlines = list(sizes.loc[matched_blobs]['midline_median'])

            self._typical_bodylength = np.median(good_midlines)

        return self._typical_bodylength
