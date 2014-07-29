#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import math

# third party
import numpy as np
import pandas as pd

# project specific
from conf import settings
from .scorer import Scorer

class Taper(object):
    """
    Initalized with a wio.Experiment-like object and a simplified graph.
    """
    def __init__(self, experiment, graph):#, regenerate_cache=False):
        self.experiment = experiment

        self._scorer = Scorer(experiment)

        self.max_speed = self._scorer.max_speed * settings.TAPE_MAX_SPEED_MULTIPLIER

        self._terminals = self.experiment.prepdata.load('terminals').set_index('bid')
        self.graph = graph

    # def score(lost_id, found_id):
    #     lost = self._terminals.loc[lost_id]
    #     found = self._terminals.loc[found_id]

    #     Dx = lost.xN - found.x0
    #     Dy = lost.yN - found.y0
    #     Dt = lost.tN - found.t0

    #     Dr = math.sqrt(Dx**2 + Dy**2)

    #     return self._scorer(Dr, Dt)

    def find_start_and_end_nodes(self):

        graph = self.graph
        terminals = self._terminals

        # go through graph and find all node ids for start/stop canidates
        start_nodes = [x for x in graph.nodes() if len(graph.successors(x)) == 0]
        stop_nodes = [x for x in graph.nodes() if len(graph.predecessors(x)) == 0]

        # reformat terminals dataframe.
        terms = terminals.copy()
        terms.reset_index(inplace=True)
        terms = terms[np.isfinite(terms['t0'])] # drop NaN rows
        term_ids = set(terms['bid']) # get set of all bids with data
        terms.set_index('bid', inplace=True)
        terms['node_id'] = 0

        # make sets that contain bids of all components
        stop_components = []
        for bid in stop_nodes:
            comps = graph[bid].get('components', [bid])
            stop_components.extend(comps)
        stop_components = set(stop_components)

        start_components = []
        for bid in start_nodes:
            comps = graph[bid].get('components', [bid])
            start_components.extend(comps)
        start_components = set(start_components)

        # print(len(term_ids), 'term ids')
        # print(len(start_components), 'start set')
        # print(len(stop_components), 'stop set')
        # print(len(stop_components & start_components), 'start/stop overlap')

        start_bids = start_components & term_ids
        stop_bids = stop_components & term_ids

        # split up terminals dataframe into two smaller ones containing
        # only relevant information.
        terms.reset_index(inplace=True)
        start_terms = terms.loc[list(start_bids)][['bid', 't0', 'x0', 'y0', 'f0', 'node_id']]
        end_terms = terms.loc[list(stop_bids)][['bid', 'tN', 'xN', 'yN', 'fN', 'node_id']]

        start_terms.rename(columns={'t0':'t', 'x0':'x', 'y0':'y',
                                    'f0':'f'}, inplace=True)

        end_terms.rename(columns={'tN':'t', 'xN':'x', 'yN':'y',
                                  'fN':'f'}, inplace=True)

        # add in the node_id values into the dataframes
        for node_id in start_nodes:
            comps = graph[node_id].get('components', [node_id])
            for comp in comps:
                if comp in start_bids:
                    start_terms['node_id'].loc[comp] = node_id

        for node_id in stop_nodes:
            comps = graph[node_id].get('components', [node_id])
            for comp in comps:
                if comp in stop_bids:
                    end_terms['node_id'].loc[comp] = node_id

        # drop rows with NaN as 't' (most other data will be missing)
        start_terms = start_terms[np.isfinite(start_terms['t'])]
        end_terms = end_terms[np.isfinite(end_terms['t'])]

        # print('dropped NaN values')
        # print(len(start_terms), 'start terms')
        # print(len(end_terms), 'end terms')

        # sort nodes using time.
        start_terms.sort(columns='t', inplace=True)
        end_terms.sort(columns='t', inplace=True)

        # drop rows that have duplicate node_ids.
        # for starts, take the first row (lowest time)
        start_terms.drop_duplicates(subset='node_id', take_last=False,
                                    inplace=True)
        # for ends, take the last row (highest time)
        end_terms.drop_duplicates(subset='node_id', take_last=True,
                                  inplace=True)

        print('removed all but one component for node_ids')
        print(len(start_terms), 'start terms')
        print(len(end_terms), 'end terms')

        return start_terms, end_terms


    def score_potential_matches(self, start_terms, end_terms):
        """

        creates a dataframe with the following columns:

        node1, node2, blob1, blob2, dist, dt, df, score

        where
        dt -- difference in seconds
        df -- difference in frames
        score -- generated from self._scorer

        params
        -----
        start_terms: (pandas DataFrame)
        end_terms: (pandas DataFrame)
        columns:
        node_id, bid, x, y, t, f

        """

        def score(row):
            #print(row)
            r = self._scorer(row['dist'], row['df'])
            return float(r)


        buffer = 10
        all_match_dfs = []
        print(end_terms.columns)

        start_num = len(start_terms)
        for row_id, row in end_terms.iterrows():
            x, y = row['x'], row['y']
            f, t = row['f'], row['t']
            node1, blob1 = row['node_id'], row['bid']

            # start narrowing down data frame by time
            match_df = start_terms.copy()
            match_df['dt'] = start_terms['t'] - t
            match_df['df'] = start_terms['f'] - f
            match_df = match_df[0 < match_df['df']]
            match_df = match_df[match_df['df'] <= 500]

            time_num = len(match_df)

            dy = start_terms['y'] - y
            dx = start_terms['x'] - x
            match_df['dist'] = np.sqrt(dy**2 + dx**2)

            speed = match_df['dist'] / match_df['df']

            match_df = match_df[ speed < (self.max_speed + buffer)]

            space_num = len(match_df)
            m = '{b}\t | {i} | {t} | {s}'.format(b=int(blob1), i=start_num,
                                                 t=time_num, s=space_num)
            #print(m)

            if len(match_df):
                # some matches were left, reformat df.
                match_df.rename(columns={'bid':'blob2',
                                         'node_id':'node2'},
                                inplace=True)
                match_df['node1'] = node1
                match_df['blob1'] = blob1
                match_df = match_df[['node1', 'node2', 'blob1', 'blob2',
                                     'dist', 'dt', 'df']]

                a = match_df[['dist', 'df']].apply(score, axis=1)
                match_df['score'] = a
                all_match_dfs.append(match_df)

        potential_matches = pd.concat(all_match_dfs)
        print(potential_matches.head())
        return potential_matches
