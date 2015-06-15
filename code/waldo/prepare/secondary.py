from __future__ import print_function, absolute_import, unicode_literals, division
"""
Using the summary data from the raw experiment, here are secondary methods
that generate other useful information as a shortcut.
"""
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library

# third party
import numpy as np
import pandas as pd

# project specific
import waldo.wio.file_manager as fm
import waldo.images as wi

def bodylengths_moved(experiment=None, bounds=None, sizes=None):
    if bounds is None or sizes is None:
        if experiment is None:
            raise ValueError('Either the experiment must be provided or '
                             'both bounds and sizes dataframes.')
        bounds = experiment.prepdata.load('bounds')
        sizes = experiment.prepdata.load('sizes')

    moved = pd.concat([bounds.set_index('bid'),
                       sizes.set_index('bid')], axis=1)
    moved.reset_index(inplace=True)

    dx = bounds['x_max'] - bounds['x_min']
    dy = bounds['y_max'] - bounds['y_min']
    box_diag = np.sqrt(dx**2 + dy**2)
    moved['bl_moved'] = box_diag / moved['midline_median']

    return moved[['bid', 'bl_moved']]

# def _check_roi(bounds, x, y, r):
#     df = bounds.copy()
#     box_x = (df['x_min'] + df['x_max']) / 2
#     box_y = (df['y_min'] + df['y_max']) / 2

#     df['inside_roi'] = r > np.sqrt((box_x - x)**2 + (box_y - y)**2)

#     return df[['bid', 'inside_roi']]

def in_roi(experiment=None, ex_id=None, bounds=None):
    """
    Generate a dataframe with True/False for each blob ID if the bounding
    box centroid lies within the ROI.

    Must provide either experiment or (ex_id and bounds). Optionally
    experiment and bounds dataframe can be provided to save some load time.
    """
    if ex_id is None or bounds is None:
        if experiment is None:
            raise ValueError('Either the experiment must be provided or '
                             'both experiment id and bounds dataframe.')
        if bounds is None:
            bounds = experiment.prepdata.load('bounds')
        if ex_id is None:
            ex_id = experiment.id

    roi_dict = fm.ImageMarkings(ex_id=ex_id).roi()
    roi_mask = wi.create_roi_mask(roi_dict)

    box_x = (bounds['x_min'] + bounds['x_max']) / 2
    box_y = (bounds['y_min'] + bounds['y_max']) / 2
    roi = wi.check_points_against_roi(box_x, box_y, roi_mask)

    return roi
