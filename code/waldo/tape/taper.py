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
import logging

# third party
import numpy as np
import pandas as pd

# project specific
from waldo.conf import settings

L = logging.getLogger(__name__)

class Taper(object):
    """
    Initalized with a wio.Experiment-like object and a simplified graph.
    """
    def __init__(self, experiment, graph, scorer=None,
                 acausal_frame_limit=None):#, regenerate_cache=False):
        self._experiment = experiment
        self._graph = graph

        if acausal_frame_limit is None:
            self.acausal_limit = settings.TAPE_ACAUSAL_FRAME_LIMIT
        else:
            self.acausal_limit = acausal_frame_limit
        self.acausal_limit = np.abs(self.acausal_limit)

        self._terminals = self._experiment.prepdata.load('terminals').set_index('bid')
        # self._missing = self._load_missing()
        self._node_start_blobs = {}
        self._node_end_blobs = {}
        self.link_history = []

    def find_start_and_end_nodes(self, use_missing_objects=False):
        graph = self._graph
        terms = self._terminals.copy()
        #print('raw terms')
        #print(terms.head(20))
        # go through graph and find all node ids for start/stop canidates
        gap_start_nodes = [x for x in graph.nodes() if len(graph.successors(x)) == 0]
        gap_end_nodes = [x for x in graph.nodes() if len(graph.predecessors(x)) == 0]

        # reformat terminals dataframe.
        terms = terms[np.isfinite(terms['t0'])] # drop NaN rows

        # make sets that contain bids of all components
        gap_start_components = set()
        for bid in gap_start_nodes:
            gap_start_components.update(graph.components(bid))

        gap_end_components = set()
        for bid in gap_end_nodes:
            gap_end_components.update(graph.components(bid))

        # print(len(term_ids), 'term ids')
        # print(len(gap_end_components), 'start set')
        # print(len(gap_start_components), 'stop set')
        # print(len(gap_start_components & gap_end_components), 'start/stop overlap')

        term_ids = set(terms.index) # get set of all bids with data
        gap_start_bids = gap_start_components & term_ids
        gap_end_bids = gap_end_components & term_ids
        isolated = gap_start_bids & gap_end_bids

        # split up terminals dataframe into two smaller ones containing
        # only relevant information.
        terms['isolated'] = False
        terms['isolated'].loc[list(isolated)]
        gap_start_terms = terms.loc[list(gap_start_bids)][['tN', 'xN', 'yN', 'fN', 'isolated']]
        gap_end_terms = terms.loc[list(gap_end_bids)][['t0', 'x0', 'y0', 'f0', 'isolated']]
        gap_start_terms.index.name = gap_end_terms.index.name = 'bid'

        gap_start_terms.rename(
                columns={'tN': 't', 'xN': 'x', 'yN': 'y', 'fN': 'f'},
                inplace=True)
        gap_end_terms.rename(
                columns={'t0': 't', 'x0': 'x', 'y0': 'y', 'f0': 'f'},
                inplace=True)

        #print('start terms1')
        #print(gap_start_terms.head(10))

        # add in the node_id values into the dataframes
        gap_start_terms['node_id'] = gap_start_terms.index
        for node_id in gap_start_nodes:
            comps = graph.components(node_id)
            for comp in comps:
                if comp in gap_start_bids:
                    gap_start_terms['node_id'].loc[comp] = node_id

        gap_end_terms['node_id'] = gap_end_terms.index

        for node_id in gap_end_nodes:
            comps = graph.components(node_id)
            # comps = graph[node_id].get('components', [node_id])
            for comp in comps:
                if comp in gap_end_bids:
                    gap_end_terms['node_id'].loc[comp] = node_id

        # drop rows with NaN as 't' (most other data will be missing)
        gap_start_terms = gap_start_terms[np.isfinite(gap_start_terms['t'])]
        gap_end_terms = gap_end_terms[np.isfinite(gap_end_terms['t'])]

        # sort nodes using time.
        gap_start_terms.sort(columns='t', inplace=True)
        gap_end_terms.sort(columns='t', inplace=True)

        # drop rows that have duplicate node_ids.
        # for gap starts, take the last row (higce=True)
        # for gap ends, take the first row (lowehest time)
        gap_start_terms.drop_duplicates(
                'node_id', take_last=True, inplace=True)
        gap_end_terms.drop_duplicates(
                'node_id', take_last=False, inplace=True)

        #print('removed all but one component for node_ids')
        #print(len(gap_end_terms), 'start terms')
        #print(len(gap_start_terms), 'end terms')

        # clean up after using index
        gap_start_terms.reset_index(inplace=True)
        gap_end_terms.reset_index(inplace=True)

        # store node - blob relations for later use.
        for i, row in gap_start_terms.iterrows():
            node, blob = row['node_id'], row['bid']
            self._node_start_blobs[int(node)] = int(blob)

        for i, row in gap_end_terms.iterrows():
            node, blob = row['node_id'], row['bid']
            self._node_end_blobs[int(node)] = int(blob)

        return gap_start_terms, gap_end_terms

    def score_potential_gaps(self, gap_start_terms, gap_end_terms,
                             preserve_history=False, write_everything=False):
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

        # pull defaults from settings
        t_buffer = self.acausal_limit
        max_distance_cutoff = settings.TAPE_PIXEL_SEARCH_LIMIT
        max_time_cuttoff = settings.TAPE_FRAME_SEARCH_LIMIT

        print('infer gaps: max dist', max_distance_cutoff)
        print('infer gaps: max time', max_time_cuttoff)

        full_record = []
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

            # must be non-isolated and have df > 0 - t_buffer
            # or
            # be isolated and have          df > 0
            gap_df = gap_df[(0 < (gap_df['df'] + t_buffer)) |
                            ((gap_df['isolated']) & (0 < gap_df['df']))]

            gap_df = gap_df[gap_df['df'] < max_time_cuttoff]

            # stop doing calculations if no canidates
            if not len(gap_df):
                continue

            time_num = len(gap_df)
            dy = gap_df['y'] - y
            dx = gap_df['x'] - x
            dist = np.sqrt(dy**2 + dx**2)
            gap_df['dist'] = dist

            if preserve_history:
                gap_df['x1'] = x
                gap_df['y1'] = y
                gap_df['t1'] = t
                gap_df['f1'] = f
                gap_df['x2'] = gap_df['x']
                gap_df['y2'] = gap_df['y']
                gap_df['t2'] = gap_df['t']
                gap_df['f2'] = gap_df['f']

            gap_df['speed'] = gap_df['dist'] / gap_df['df']
            if write_everything:
                full_record.append(gap_df)

            gap_df = gap_df[gap_df['dist'] < max_distance_cutoff]

            # remove self links if we allow short backwards links.
            if t_buffer > 0:
                gap_df = gap_df[gap_df['node_id'] != node1]

            space_num = len(gap_df)
            m = '{b}\t | {i} | {t} | {s}'.format(b=blob1, i=start_num,
                                                 t=time_num, s=space_num)
            if len(gap_df):
                # some gaps were left, reformat df.
                gap_df.rename(columns={'bid':'blob2',
                                       'node_id':'node2'},
                                inplace=True)
                gap_df['node1'] = node1
                gap_df['blob1'] = blob1
                if not preserve_history:
                    gap_df = gap_df[['node1', 'node2', 'blob1', 'blob2',
                                     'dist', 'dt', 'df']]
                else:
                    gap_df = gap_df[['node1', 'node2', 'blob1', 'blob2',
                                     'x1', 'x2', 'y1', 'y2',
                                     'f1', 'f2', 't1', 't2',
                                     'dist', 'dt', 'df']]

                # a = gap_df[['dist', 'df']].apply(score, axis=1)
                # gap_df['score'] = a
                all_gap_dfs.append(gap_df)

        any_data = [d for d in all_gap_dfs if d is not None]
        if not any_data:
            return None

        potential_gaps = pd.concat(all_gap_dfs)
        if write_everything:
            all_gaps = pd.concat(full_record)
            all_gaps.to_csv('all_gaps.csv')
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

    def short_tape(self, gaps_df, df=None, dist=None, add_edges=True):
        """
        preferentially attaches really short gaps to one another.

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
        if df is None:
            df = settings.TAPE_FRAME_SEARCH_LIMIT
        if dist is None:
            dist = settings.TAPE_PIXEL_SEARCH_LIMIT

        acausal_limit = self.acausal_limit

        if gaps_df is None:
            # if there is no gaps file, then there is no point in trying to find gaps
            print('WARNING: no gaps file')
            return [], []

        gaps = gaps_df.copy()
        link_list = []
        gaps = gaps[gaps['df'] < df]
        gaps = gaps[gaps['dist'] < dist]
        gaps = gaps[gaps['df'] > - acausal_limit]

        # rank all gaps based on a score
        gaps['short_score'] = gaps['dt'] * gaps['dist']
        gaps.sort('short_score', inplace=True, ascending=True)
        already_taken_nodes = set() #set of nodes already involved
        for i, row in gaps.iterrows():
            node1, node2 = row['node1'], row['node2']
            # skip nodes if they are already used in a gap
            if node1 in already_taken_nodes:
                continue
            if node2 in already_taken_nodes:
                continue
            # add if gap is less than distance or time requirements
            if row['df'] <= df and row['dist'] <= dist:
                link_list.append((node1, node2))
                already_taken_nodes.add(node1)
                already_taken_nodes.add(node2)
                print('gaps getting joined:')
                print(link_list)

        if add_edges:
            graph = self._graph

            link_history = [(self._node_start_blobs[n1],
                        self._node_end_blobs[n2])
                        for (n1, n2) in link_list]
            self.link_history.extend(link_history)
            success = link_list

            self._add_taped_edges_to_graph(link_list)
        return success, gaps


    def _add_taped_edges_to_graph(self, link_list):
        """
        Adds edges from *link_list* to the graph. All added edges have the
        'taped' property set True.
        """

        #self._graph.add_edges_from(real_to_real, taped=True)


        blob_to_blob = [(self._node_start_blobs[n1],
                            self._node_end_blobs[n2])
                        for (n1, n2) in link_list]

        self._graph.bridge_gaps(link_list, blob_to_blob)
