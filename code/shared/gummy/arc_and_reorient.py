# -*- coding: utf-8 -*-
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

from .synthetic import SyntheticWorm

import numpy as np
from scipy.stats import expon


class ArcAndReorient(SyntheticWorm):
    """
    Worms that truck forward while gradually arcing.  Randomly change speeds
    and if stopped, can randomly reorient along any heading.
    """

    REALIZATION_DEFAULTS = {
        'speeds': [0, 0.01, 0.04],
        'angle_dist_mean': 0,
        'angle_dist_sd': 0.01,
        'reorient_chance': 0.1,
        'expon_rvs_scale': 100,
    }

    def realization(self):
        """
        Generate and return an ideal realization.
        """
        seg_len_dist = expon(scale=self.params['expon_rvs_scale'])

        state = (0, 0, 0, 0)
        states = [state]

        while len(states) < self.n_points + 2:
            # intialize velocity and angular change
            new_v = np.random.choice(self.params['speeds'])
            d_ang = 0
            if new_v > 0:
                d_ang = np.random.normal(
                        self.params['angle_dist_mean'],
                        self.params['angle_dist_sd'])

            # before moving, try reorienting
            if new_v > 0 and np.random.uniform() < self.params['reorient_chance']:
                # choose a completely random direction
                x, y, v, ang = states[-1]
                states[-1] = (x, y, v, np.random.uniform(-np.pi, np.pi))

            # now generate all changes.
            seg_len = int(seg_len_dist.rvs())
            x, y, v, ang = states[-1]
            for i in range(seg_len):
                x, y, v, ang = (
                        x + v * np.cos(ang),
                        y + v * np.sin(ang),
                        new_v,
                        ang + d_ang)
                states.append((x, y, v, ang))

        output = np.array(states[2:self.n_points+2])

        return output[...,:2]
