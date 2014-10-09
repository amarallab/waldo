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
        self.f = None
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
            trackframes_pm = zip(self.trackframes_p[track_index], self.trackframes_m[track_index])
            for frame_index, (tp, tm) in enumerate(trackframes_pm):
                p = IdTrackSolver._best_distance(tp, self.reference_p)
                m = IdTrackSolver._best_distance(tm, self.reference_m)
                a = -1 if p != m else p
                data_p[track_index, frame_index] = p
                data_m[track_index, frame_index] = m
                agreement[track_index, frame_index] = a

        self.track_assignment_p = data_p
        self.track_assignment_m = data_m
        self.agreement = agreement

    def debug_create_test_data(self):
        track_size=4
        frameset_size=10
        wids_size=3
        self.track_ids = np.array(range(track_size)) + 1
        self.wids = np.array(range(wids_size)) + 1

        data_p = np.zeros((track_size, frameset_size), dtype=int)
        data_m = np.zeros((track_size, frameset_size), dtype=int)
        agreement = np.zeros((track_size, frameset_size), dtype=int)
        for track_index, track_in in enumerate(self.track_ids):
            for frame_index in range(frameset_size):
                p = np.random.random_integers(0, wids_size - 1) # index, not value
                if np.random.random() < 0.6:
                    m = p
                else:
                    m = np.random.random_integers(wids_size - 1)
                a = -1 if p != m else p
                data_p[track_index, frame_index] = p
                data_m[track_index, frame_index] = m
                agreement[track_index, frame_index] = a
        self.track_assignment_p = data_p
        self.track_assignment_m = data_m
        self.agreement = agreement

    @staticmethod
    def compute_weight(image, stack):
        # p: number of pixels of the blob (image)
        # b: number of blobs of the fragment that overlap with that same pixel
        # bb: matrix of 'b' (for each pixel)
        p = image.sum()
        bb = sum([image * s for s in stack])
        pbb = p * bb
        inv = (1 / pbb)
        inv[np.isinf(inv)] = 0
        return inv.sum()

    def add_frame_weights(self):
        track_size = len(self.track_ids)
        frameset_size = len(self.trackframes_p[0])
        frame_weights = np.zeros((track_size, frameset_size)) + 1
        #TODO: What?????
        self.frame_weights = frame_weights

    def compute_f(self):
        f = np.zeros((len(self.track_ids), len(self.wids)))
        for track_index, track_id in enumerate(self.track_ids):
            for wid_index, wid in enumerate(self.wids):
                ta = self.agreement[track_index, :]
                tw = self.frame_weights[track_index, :]
                f[track_index, wid_index] = np.sum(tw[ta == wid_index])
        self.f = f

    def compute_p1(self):
        p1 = np.zeros((len(self.track_ids), len(self.wids)))
        for track_index, track_id in enumerate(self.track_ids):
            den = sum(np.power(2, self.f[track_index, :]))
            if den != 0:
                for wid_index, wid in enumerate(self.wids):
                    num = np.power(2, self.f[track_index, wid_index])
                    p1[track_index, wid_index] = num / den

        # TESTING: create a random matrix to test p2
        # test_size = 3
        # p1 = np.random.random(test_size * test_size)
        # p1.shape = (test_size, test_size)
        # self.wids = self.wids[0:test_size]
        # self.track_ids = self.track_ids[0:test_size]
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
