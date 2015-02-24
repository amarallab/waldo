from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import random

import unittest
from nose.tools import nottest
import networkx as nx
import pandas as pd

#from .test_util import node_generate, cumulative_sum, GraphCheck
from waldo.network.tests import test_graph as tg

from .. import Taper

class FakePrepdata(object):
    def __init__(self, terminals, missing):
        self.terminals = terminals
        self.missing = missing

        if missing is None:
            self.missing = self.default_missing()

    def load(self, data_id):
        if data_id == 'terminals':
           return self.terminals
        elif data_id == 'missing':
           return self.missing
        else:
            print('error. that type of data not in FakePrepdata')

    def default_missing(self):
        columns = ['id','F','t','x','y','xmin','ymin',
                   'xmax','ymax','next']
        data = [['m0',1.0,0.0,276.5599547511312,317.10972850678735,
                 249.0,284.0,306.0,349.0,None],
                ['m1',1.0,0.0,921.7716894977169,1234.132420091324,
                 903.0,1225.0,946.0,1243.0,None]]
        return pd.DataFrame(data, columns=columns)

#columns for terminals
#['bid','x0','y0','t0','f0','xN','yN','tN','fN']

class FakeExperiment(object):
    def __init__(self, graph, terminals=None, missing=None):
        self.graph = graph
        terms = []

        for bid, blob_data in graph.nodes(data=True):
            d = terminals[bid]
            d['bid'] = bid
            d['t0'] = d.get('t0', blob_data['born_t'])
            d['tN'] = d.get('tN', blob_data['died_t'])
            d['f0'] = d.get('f0', blob_data['born_f'])
            d['fN'] = d.get('fN', blob_data['died_f'])
            terms.append(d)

            comps = blob_data.get('components', [])
            for c in comps:
                d = terminals[c]
                if 't0' not in d:
                    print('WARNING need to include times while'
                          'defining component terminals')
                d['bid'] = c
                terms.append(d)

        self.terminals = pd.DataFrame(terms)
        self.prepdata = FakePrepdata(self.terminals, missing)

class FakeScorer(object):
    def __init__(self, terminals=None, missing=None):
        self.max_speed = 10
        self._interpolator = None

    def __call__(self, frame_gap, distance_gap):
        if self._interpolator is None:
            return 5

        return float(max(self._interpolator(frame_gap, distance_gap), 1e-100))

def df_equal( df1, df2 ):
    """ Check if two DataFrames are equal, ignoring nans """
    return df1.fillna(1).sort(axis=1).eq(df2.fillna(1).sort(axis=1)).all().all()

class TestTaper(unittest.TestCase):

    @nottest
    def test_basic_pass(self):
        # taper settings
        max_df = 6
        max_dist = 4

        # the end of the source node
        x = 300
        y = 200
        f = 100

        # the relative position of the start of the sink node
        df = 5
        dx = 1
        dy = 1

        Go = tg.node_generate([[10], [], [20]],
                              [1, f, f + df, 300])
        terms = {}
        terms[10] = {'x0': 1, 'y0':201 , 'xN': x, 'yN': y,}
        terms[20] = {'x0': x + dx, 'y0':y + dy, 'xN': 301, 'yN':501,}

        expected_pairs = {10: 20}

        experiment = FakeExperiment(graph=Go, terminals=terms)
        scorer = FakeScorer()

        taper = Taper(graph=Go, experiment=experiment, scorer=scorer)
        gap_start, gap_end = taper.find_start_and_end_nodes()
        gaps = taper.score_potential_gaps(gap_start, gap_end)
        link_list, gaps = taper.short_tape(gaps,
                                           df=max_df,
                                           dist=max_dist)
        #print(expected_pairs)
        #print(link_list, len(link_list))
        self.assertTrue(len(link_list) == len(expected_pairs))
        for s, e in link_list:
            self.assertTrue(s in expected_pairs)
            self.assertTrue(expected_pairs[s] == e)

    @nottest
    def test_basic_dy_fail(self):
        # the relative position of the start of the sink node
        df = 5
        dx = 0
        dy = 6.1

        # taper settings
        max_df = 6
        max_dist = 6

        expected_pairs = {}
        # the end of the source node
        x = 300
        y = 200
        f = 100


        Go = tg.node_generate([[10], [], [20]], [1, f, f + df, 300])

        terms = {}
        terms[10] = {'x0': 1, 'y0':201 , 'xN': x, 'yN': y,}
        terms[20] = {'x0': x + dx, 'y0':y + dy, 'xN': 301, 'yN':501,}

        experiment = FakeExperiment(graph=Go, terminals=terms)
        scorer = FakeScorer()

        taper = Taper(graph=Go, experiment=experiment, scorer=scorer)
        gap_start, gap_end = taper.find_start_and_end_nodes()
        gaps = taper.score_potential_gaps(gap_start, gap_end)
        link_list, gaps = taper.short_tape(gaps,
                                           df=max_df,
                                           dist=max_dist)
        #print(link_list, len(link_list))
        self.assertTrue(len(link_list) == len(expected_pairs))
        for s, e in link_list:
            self.assertTrue(s in expected_pairs)
            self.assertTrue(expected_pairs[s] == e)

    @nottest
    def test_basic_dx_fail(self):
        # the relative position of the start of the sink node
        df = 5
        dx = 6.1
        dy = 0

        # taper settings
        max_df = 6
        max_dist = 6

        expected_pairs = {}
        # the end of the source node
        x = 300
        y = 200
        f = 100


        Go = tg.node_generate([[10], [], [20]], [1, f, f + df, 300])

        terms = {}
        terms[10] = {'x0': 1, 'y0':201 , 'xN': x, 'yN': y,}
        terms[20] = {'x0': x + dx, 'y0':y + dy, 'xN': 301, 'yN':501,}


        experiment = FakeExperiment(graph=Go, terminals=terms)
        scorer = FakeScorer()

        taper = Taper(graph=Go, experiment=experiment, scorer=scorer)
        gap_start, gap_end = taper.find_start_and_end_nodes()
        gaps = taper.score_potential_gaps(gap_start, gap_end)
        link_list, gaps = taper.short_tape(gaps,
                                           df=max_df,
                                           dist=max_dist)
        #print(link_list, len(link_list))
        self.assertTrue(len(link_list) == len(expected_pairs))
        for s, e in link_list:
            self.assertTrue(s in expected_pairs)
            self.assertTrue(expected_pairs[s] == e)

    @nottest
    def test_basic_dt_fail(self):
        # the relative position of the start of the sink node
        df = 6
        dx = 0
        dy = 0

        # taper settings
        max_df = 6
        max_dist = 6

        expected_pairs = {}
        # the end of the source node
        x = 300
        y = 200
        f = 100

        Go = tg.node_generate([[10], [], [20]], [1, f, f + df, 300])

        terms = {}
        terms[10] = {'x0': 1, 'y0':201 , 'xN': x, 'yN': y,}
        terms[20] = {'x0': x + dx, 'y0':y + dy, 'xN': 301, 'yN':501,}


        experiment = FakeExperiment(graph=Go, terminals=terms)
        scorer = FakeScorer()

        taper = Taper(graph=Go, experiment=experiment, scorer=scorer)
        gap_start, gap_end = taper.find_start_and_end_nodes()
        gaps = taper.score_potential_gaps(gap_start, gap_end)
        link_list, gaps = taper.short_tape(gaps,
                                           df=max_df,
                                           dist=max_dist)
        #print(link_list, len(link_list))
        self.assertTrue(len(link_list) == len(expected_pairs))
        for s, e in link_list:
            self.assertTrue(s in expected_pairs)
            self.assertTrue(expected_pairs[s] == e)

    @nottest
    def test_basic_acausal_pass(self):
        # the relative position of the start of the sink node
        df = -20
        dx = 1
        dy = 1

        expected_pairs = {10: 20}

        # taper settings
        max_df = 6
        max_dist = 20
        acausal_limit = -30

        # the end of the source node
        x = 300
        y = 200
        f = 100

        Go = tg.node_generate([[10], [], [20]], [1, f, f + df, 300])

        terms = {}
        terms[10] = {'x0': 1, 'y0':201 , 'xN': x, 'yN': y,}
        terms[20] = {'x0': x + dx, 'y0':y + dy, 'xN': 301, 'yN':501,}

        experiment = FakeExperiment(graph=Go, terminals=terms)
        scorer = FakeScorer()

        taper = Taper(graph=Go, experiment=experiment, scorer=scorer,
                      acausal_frame_limit=acausal_limit)
        gap_start, gap_end = taper.find_start_and_end_nodes()
        gaps = taper.score_potential_gaps(gap_start, gap_end)
        link_list, gaps = taper.short_tape(gaps,
                                           df=max_df,
                                           dist=max_dist)
        #print(link_list, len(link_list))
        self.assertTrue(len(link_list) == len(expected_pairs))
        for s, e in link_list:
            self.assertTrue(s in expected_pairs)
            self.assertTrue(expected_pairs[s] == e)
    @nottest
    def test_basic_acausal_fail(self):
        # the relative position of the start of the sink node
        df = -3
        dx = 1
        dy = 1

        # taper settings
        max_df = 6
        max_dist = 6
        acausal_limit = -2

        expected_pairs = {}
        # the end of the source node
        x = 300
        y = 200
        f = 100

        Go = tg.node_generate([[10], [], [20]], [1, f, f + df, 300])

        terms = {}

        terms[10] = {'x0': 1, 'y0':201 , 'xN': x, 'yN': y,}
        terms[20] = {'x0': x + dx, 'y0':y + dy, 'xN': 301, 'yN':501,}

        experiment = FakeExperiment(graph=Go, terminals=terms)
        scorer = FakeScorer()

        taper = Taper(graph=Go, experiment=experiment, scorer=scorer,
                      acausal_frame_limit=acausal_limit)
        gap_start, gap_end = taper.find_start_and_end_nodes()
        gaps = taper.score_potential_gaps(gap_start, gap_end)
        self.assertTrue(gaps is None)

    def test_component_terminal_selection(self):
        # the relative position of the start of the sink node
        df = 20
        dx = 1
        dy = 1

        expected_pairs = {10: 20}

        # taper settings
        max_df = 6
        max_dist = 20
        acausal_limit = -3

        # the end of the source node
        x = 300
        y = 200
        f = 100

        # component intersection
        cx = 300
        cy = 200
        cf = 100

        Go = tg.node_generate([[10], [], [20]], [1, f, f + df, 300])
        # TODO add a component to 10

        terms = {}
        terms[10] = {'x0': 1, 'y0':201 , 'xN': cx, 'yN': cy,
                     't0': 0.1, 'f0':1, 'tN': cf * 0.1, 'fN': cf}

        terms[11] = {'x0': cx + 5, 'y0': cy - 5 , 'xN': x, 'yN': y,
                     't0': 0.1, 'f0':1, 'tN': cf * 0.1, 'fN': cf}

        terms[20] = {'x0': x + dx, 'y0':x + dy, 'xN': 301, 'yN':501,}

        experiment = FakeExperiment(graph=Go, terminals=terms)
