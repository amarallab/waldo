import os
import sys
import pandas as pd
import numpy as np
import prettyplotlib as ppl
import matplotlib.pyplot as plt
from glob import glob
import h5py
import collections


class IdTrackSolver(object):


    def __init__(self, wids, reference_p, reference_m):
        self.wids = wids
        self.reference_m = reference_m
        self.reference_p = reference_p

        self.track_ids = None
        self.test_p = None
        self.test_m = None
        self.frame_weights = None
        self.p1 = None
        self.p2 = None


    def insert_test_tracks(self, test_p, test_m, track_ids=None):
        self.test_p = test_p
        self.test_m = test_m

        if track_ids is None:
            track_ids = self.wids
        self.track_ids = track_ids

    def initial_assignment(self):
        pass

    def add_frame_weights(self):
        pass

    def compute_p1(self):
        pass

    def compute_p2(self):
        pass
