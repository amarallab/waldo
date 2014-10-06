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
        self.trackframes_p = None
        self.trackframes_m = None
        self.track_assignment_p = None
        self.track_assignment_m = None
        self.agreement = None
        self.frame_weights = None
        self.p1 = None
        self.p2 = None


    def insert_trackframes(self, trackframes_p, trackframes_m, track_ids=None):
        self.trackframes_p = trackframes_p
        self.trackframes_m = trackframes_m

        self.track_ids = track_ids if track_ids is not None else self.wids

    @staticmethod
    def _best_distance(frame, references):
        best_index, best_distance = None, None
        for index, current in enumerate(references):
            values = np.fabs(frame - current)
            if len(values) == 0:
                return -1
            distance = np.min(values.mean(axis=1).mean(axis=1))
            if best_index is None or distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def initial_assignment(self):
        track_size = len(self.track_ids)
        frameset_size = len(self.trackframes_p[0])
        data_p = np.zeros((track_size, frameset_size), dtype=int)
        data_m = np.zeros((track_size, frameset_size), dtype=int)
        agreement = np.zeros((track_size, frameset_size), dtype=int)
        for track_index, track_id in enumerate(self.track_ids):
            for frame_index, (tp, tm) in enumerate(zip(self.trackframes_p[track_index], self.trackframes_m[track_index])):
                p = IdTrackSolver._best_distance(tp, self.reference_p)
                m = IdTrackSolver._best_distance(tm, self.reference_m)
                a = -1 if p != m else p
                data_p[track_index, frame_index] = p
                data_m[track_index, frame_index] = m
                agreement[track_index, frame_index] = a

        self.track_assignment_p = data_p
        self.track_assignment_m = data_m
        self.agreement = agreement

    def add_frame_weights(self):
        pass

    def compute_p1(self):
        p1 = np.zeros((len(self.track_ids), len(self.wids)))
        for track_index, track_id in enumerate(self.track_ids):
            f = np.zeros(len(self.wids))
            for wid_index, wid in enumerate(self.wids):
                f[wid_index] = np.power(2, len([x for x in self.agreement[track_index, :] if x == wid_index]))

            sum = f.sum()
            if sum != 0:
                for wid_index, wid in enumerate(self.wids):
                    p1[track_index, wid_index] = f[wid_index] / f.sum()
        self.p1 = p1

    def _compute_p2_components_for_track(self, in_track_index):
        components = np.zeros(len(self.wids))
        for wid_index, wid in enumerate(self.wids):
            current = 1
            for track_index, track_id in enumerate(self.track_ids):
                value = self.p1[track_index, wid_index]
                if track_index == in_track_index:
                    current *= value
                else:
                    current *= 1 - value
            components[wid_index] = current
        return components

    def compute_p2(self):
        p2 = np.zeros((len(self.track_ids), len(self.wids)))
        for track_index, track_id in enumerate(self.track_ids):
            components = self._compute_p2_components_for_track(track_index)
            den = components.sum()
            if den == 0:
                for wid_index, wid in enumerate(self.wids):
                    p2[track_index, wid_index] = 0
            else:
                for wid_index, wid in enumerate(self.wids):
                    p2[track_index, wid_index] = components[wid_index] / den
        self.p2 = p2
