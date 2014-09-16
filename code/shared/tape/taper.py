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
        self._experiment = experiment
        self._graph = graph

        self._scorer = Scorer(experiment)

        self.max_speed = self._scorer.max_speed * settings.TAPE_MAX_SPEED_MULTIPLIER

        self._terminals = self._experiment.prepdata.load('terminals').set_index('bid')

    def score(distance_gap=None, frame_gap=None, ids=None):
        """
        Score the putative link based on either the distance and frame gap
        or a pair of IDs.
        """
        if distance_gap is not None and frame_gap is not None:
            Dr = distance_gap
            Df = frame_gap

        else:
            lost_id, found_id = ids

            lost = self._terminals.loc[lost_id]
            found = self._terminals.loc[found_id]

            Dx = lost.xN - found.x0
            Dy = lost.yN - found.y0
            Dr = math.sqrt(Dx**2 + Dy**2)
            Df = lost.fN - found.f0

        return self._scorer(Dr, Df)

    def find_start_and_end_nodes(self):

        graph = self._graph
        terminals = self._terminals

        # go through graph and find all node ids for start/stop canidates
        gap_end_nodes = [x for x in graph.nodes() if len(graph.predecessors(x)) == 0]
        gap_start_nodes = [x for x in graph.nodes() if len(graph.successors(x)) == 0]

        # reformat terminals dataframe.
        terms = terminals.copy()
        terms.reset_index(inplace=True)
        terms = terms[np.isfinite(terms['t0'])] # drop NaN rows
        terms.set_index('bid', inplace=True)
        terms['node_id'] = 0

        # make sets that contain bids of all components
        gap_start_components = []
        for bid in gap_start_nodes:
            comps = graph[bid].get('components', [bid])
            gap_start_components.extend(comps)
        gap_start_components = set(gap_start_components)

        gap_end_components = []
        for bid in gap_end_nodes:
            comps = graph[bid].get('components', [bid])
            gap_end_components.extend(comps)
        gap_end_components = set(gap_end_components)

        # print(len(term_ids), 'term ids')
        # print(len(gap_end_components), 'start set')
        # print(len(gap_start_components), 'stop set')
        # print(len(gap_start_components & gap_end_components), 'start/stop overlap')

        term_ids = set(terms.index) # get set of all bids with data
        gap_end_bids = gap_end_components & term_ids
        gap_start_bids = gap_start_components & term_ids

        # split up terminals dataframe into two smaller ones containing
        # only relevant information.
        terms.reset_index(inplace=True)
        gap_end_terms = terms.loc[list(gap_end_bids)][['bid', 't0', 'x0', 'y0', 'f0', 'node_id']]
        gap_start_terms = terms.loc[list(gap_start_bids)][['bid', 'tN', 'xN', 'yN', 'fN', 'node_id']]

        gap_end_terms.rename(columns={'t0':'t', 'x0':'x', 'y0':'y',
                                    'f0':'f'}, inplace=True)

        gap_start_terms.rename(columns={'tN':'t', 'xN':'x', 'yN':'y',
                                  'fN':'f'}, inplace=True)

        # add in the node_id values into the dataframes
        for node_id in gap_end_nodes:
            comps = graph[node_id].get('components', [node_id])
            for comp in comps:
                if comp in gap_end_bids:
                    gap_end_terms['node_id'].loc[comp] = node_id

        for node_id in gap_start_nodes:
            comps = graph[node_id].get('components', [node_id])
            for comp in comps:
                if comp in gap_start_bids:
                    gap_start_terms['node_id'].loc[comp] = node_id

        # drop rows with NaN as 't' (most other data will be missing)
        gap_end_terms = gap_end_terms[np.isfinite(gap_end_terms['t'])]
        gap_start_terms = gap_start_terms[np.isfinite(gap_start_terms['t'])]

        # print('dropped NaN values')
        # print(len(gap_end_terms), 'start terms')
        # print(len(gap_start_terms), 'end terms')

        # sort nodes using time.
        gap_end_terms.sort(columns='t', inplace=True)
        gap_start_terms.sort(columns='t', inplace=True)

        # drop rows that have duplicate node_ids.
        # for starts, take the first row (lowest time)
        gap_end_terms.drop_duplicates('node_id', take_last=False,
                                    inplace=True)
        # for ends, take the last row (highest time)
        gap_start_terms.drop_duplicates('node_id', take_last=True,
                                  inplace=True)

        #print('removed all but one component for node_ids')
        #print(len(gap_end_terms), 'start terms')
        #print(len(gap_start_terms), 'end terms')

        return gap_start_terms, gap_end_terms


    def score_potential_gaps(self, gap_start_terms, gap_end_terms):
        """

        creates a dataframe with the following columns:

        node1, node2, blob1, blob2, dist, dt, df, score

        where
        dt -- difference in seconds
        df -- difference in frames
        score -- generated from self._scorer

        params
        -----
        gap_start_terms: (pandas DataFrame)
        gap_end_terms: (pandas DataFrame)

        *gap_end_terms* and *gap_start_terms* must have columns with: node_id, bid,
        x, y, t, f

        """
        def score(row):
            #print(row)
            s = self._scorer(row['df'], row['dist'])
            return s

        buffer = 10
        all_gap_dfs = []
        #print(gap_start_terms.columns)

        start_num = len(gap_end_terms)
        for row_id, row in gap_start_terms.iterrows():
            x, y = row['x'], row['y']
            f, t = row['f'], row['t']
            node1, blob1 = row['node_id'], row['bid']

            # start narrowing down data frame by time
            gap_df = gap_end_terms.copy()
            gap_df['dt'] = gap_end_terms['t'] - t
            gap_df['df'] = gap_end_terms['f'] - f
            gap_df = gap_df[
                    (0 < gap_df['df']) &
                    (gap_df['df'] <= settings.TAPE_FRAME_SEARCH_LIMIT)
                ]

            # stop doing calculations if no canidates
            if not len(gap_df):
                continue

            time_num = len(gap_df)
            dy = gap_df['y'] - y
            dx = gap_df['x'] - x
            dist = np.sqrt(dy**2 + dx**2)
            gap_df['dist'] = dist


            speed = gap_df['dist'] / gap_df['df']
            gap_df = gap_df[ speed < (self.max_speed + buffer)]

            space_num = len(gap_df)
            m = '{b}\t | {i} | {t} | {s}'.format(b=int(blob1), i=start_num,
                                                 t=time_num, s=space_num)
            #print(m)

            if len(gap_df):
                # some gaps were left, reformat df.
                gap_df.rename(columns={'bid':'blob2',
                                         'node_id':'node2'},
                                inplace=True)
                gap_df['node1'] = node1
                gap_df['blob1'] = blob1
                gap_df = gap_df[['node1', 'node2', 'blob1', 'blob2',
                                     'dist', 'dt', 'df']]

                a = gap_df[['dist', 'df']].apply(score, axis=1)
                gap_df['score'] = a
                all_gap_dfs.append(gap_df)

        potential_gaps = pd.concat(all_gap_dfs)
        #print(potential_gaps.head())
        return potential_gaps

    def make_gaps_file(self):
        s, e = self.find_start_and_end_nodes()
        gaps = self.score_potential_gaps(s, e)

        # clean up output floating point
        gaps[['node1', 'node2', 'blob1', 'blob2', 'df']] = (
            gaps[['node1', 'node2', 'blob1', 'blob2', 'df']].astype(int))
        gaps['dt'] = gaps['dt'].apply(lambda x: format(x, '.03f'))
        gaps['dist'] = gaps['dist'].apply(lambda x: format(x, '.01f'))

        self._experiment.prepdata.dump('gaps', gaps, index=False)

    def greedy_tape(self, gaps_df, threshold=0.1, add_edges=True):
        """
        Uses a greedy algorithm is to add edges (that score above a
        threshold) into the directed graph.

        if a edge is drawn between node1 and node2, then node1 is
        removed from the pool of track ends and node2 is removed from
        the pool of track starts.

        This ensures that each start and each end are only matched to
        one other node.

        this process is repeated until there are no longer any gaps
        that score above the threshold.


        params
        ----
        gaps_df: (pandas dataframe)
            dataframe generated by self.score potential gaps
        threshold: (float)
            scores must be greater than this number

        returns
        -----
        link_list: (list of tuples)
            the list of (node1 --> node2) tuples that were joined.
        """

        gaps = gaps_df.copy()
        link_list = []

        gaps.sort('score', inplace=True, ascending=False)
        above_threshold = True
        while above_threshold:
            row = gaps.iloc[0]
            score = row['score']
            node1, node2 = row['node1'], row['node2']
            above_threshold = score > threshold

            if above_threshold:
                link_list.append((node1, node2))
                gaps = gaps[gaps['node1'] != node1]
                gaps = gaps[gaps['node2'] != node2]
        if add_edges:
            self._add_taped_edges_to_graph(link_list)
        return link_list

    def lazy_tape(self, gaps_df, threshold=0.1, add_edges=True):
        """
        Adds all edges that score above a threshold into the directed
        graph. There are no controls placed on how many edges are drawn
        to each node.

        params
        ----
        gaps_df: (pandas dataframe)
            dataframe generated by self.score potential gaps
        threshold: (float)
            scores must be greater than this number

        returns
        -----
        link_list: (list of tuples)
            the list of (node1 --> node2) tuples that were joined.
        """
        gaps = gaps_df.copy()
        link_list = []
        gaps = gaps[gaps['score'] > threshold]
        link_list = list(zip(gaps['node1'], gaps['node2']))
        if add_edges:
            self._add_taped_edges_to_graph(link_list)
        return link_list

    def _add_taped_edges_to_graph(self, link_list):
        """
        adds edges to the self.digraph. all added edges have a
        'taped': True attribute.

        params
        -----
        link_list: (list of tuples)

        """
        #print(len(link_list))
        #print('starting edge number', self._graph.number_of_edges())
        self._graph.add_edges_from((n1, n2) for (n1, n2) in link_list)
        #print('after edge number', self._graph.number_of_edges())
