import unittest
import itertools

import networkx as nx

from waldo.network import Graph
from waldo.network import keyconsts as kc

FRAME_TIME = 0.1

def node_generate(nodes, timepoints=None, graph=None):
    """
    *nodes* is an iterable of iterables that define what nodes to create
    spanning each gap in *timepoints*, an iterable of numbers.  *timepoints*
    must be 1 longer than nodes.
    """
    if timepoints is None:
        timepoints = itertools.count(start=100, step=100)

    try:
        if len(nodes) != len(timepoints) - 1:
            raise ValueError('nodes is not one less in length than timepoints')
    except TypeError:
        # assuming timepoints is a generator which may or may not be
        # limitless
        pass

    if graph is None:
        graph = Graph()
    timepoints = iter(timepoints)

    birth = next(timepoints)
    for node_group, death in zip(nodes, timepoints):
        graph.add_nodes_from(node_group, born_f=birth, died_f=death, born_t=birth*FRAME_TIME, died_t=death*FRAME_TIME)
        birth = death

    return graph

def diamond_graph_a(tag=False):
    nodes = [
        [1, 2],
        [3],
        [4, 5],
        [6],
        [7, 8],
    ]

    Gtest = node_generate(nodes)
    Gtest.add_path([1, 3, 4, 6, 7])
    Gtest.add_path([2, 3, 5, 6, 8])

    if tag:
        Gtest.tag_edges()

    return Gtest

def graph_b(tag=False):
    nodes = [
        [11, 12],
        [21],
        [31, 32],
        [41],
        [51, 52],
        [61],
        [71, 72],
    ]

    Gtest = node_generate(nodes)
    Gtest.add_path([11, 21, 31, 41, 51, 61, 71])
    Gtest.add_path([12, 21, 32, 41, 52, 61, 72])

    if tag:
        Gtest.tag_edges()

    return Gtest


class GraphTestCase(unittest.TestCase):
    def assertTopologyEqual(self, Gtest, Gexpect):
        # # check topologies
        nGt = set(Gtest.nodes_iter())
        nGe = set(Gexpect.nodes_iter())
        eGt = set(Gtest.edges_iter())
        eGe = set(Gexpect.edges_iter())

        msg_lines = []
        def msg_if_things(title, things):
            if things:
                msg_lines.append('{:.<25} {}'.format(
                        title, ', '.join(str(x) for x in things)))

        msg_if_things('Unexpected nodes', nGt - nGe)
        msg_if_things('Missing nodes', nGe - nGt)
        msg_if_things('Unexpected edges', eGt - eGe)
        msg_if_things('Missing edges', eGe - eGt)

        msg = '\n'.join(msg_lines)

        self.assertTrue(nGt == nGe and eGt == eGe,
                        'Graph mismatch.\n\n{}'.format(msg))

    def assertNodeDataEquals(self, key, G, n, expected_value):
        try:
            self.assertEqual(G.node[n][key], expected_value)
        except KeyError as e:
            if str(e) == repr(n):
                self.fail('Graph missing expected node "{}"'.format(n))
            elif str(e) == repr(key):
                self.fail('Node data missing expected "{}" key'.format(key))
            else:
                raise e

    def assertNodeComponentInfo(self, *args, **kwargs):
        return self.assertNodeDataEquals(kc.COMPONENTS, *args, **kwargs)

    def assertNodeGapInfo(self, *args, **kwargs):
        return self.assertNodeDataEquals(kc.TAPED, *args, **kwargs)

    def assertEdgeDataEquals(self, key, G, edge, expected_value):
        data = G.get_edge_data(*edge)
        if data is None:
            self.fail('Graph missing edge {}-{}'.format(*edge))
        try:
            self.assertEqual(data[key], expected_value)
        except KeyError as e:
            if str(e) == repr(key):
                self.fail('Edge data missing expected "{}" key'.format(key))
            else:
                raise e

    def assertEdgeInfo(self, Gtest, edge_tags):
        for a, b, data in Gtest.edges_iter(data=True):
            try:
                edgeids = data[kc.BLOB_ID_EDGES]
            except KeyError:
                self.fail('Data for edge {} -> {} does not have key "{}"'
                        .format(a, b, kc.BLOB_ID_EDGES))

            try:
                expected = edge_tags[(a, b)]
            except KeyError:
                self.fail('Unexpected edge found: {} -> {}. Expected edges: {}'
                        .format(a, b, list(edge_tags.keys())))

            self.assertEqual(edgeids, expected)


class GraphTestCaseTest(GraphTestCase):
    def test_check_ok(self):
        A = Graph()
        B = Graph()

        A.add_path([1, 2, 3])
        B.add_path([1, 2, 3])

        self.assertTopologyEqual(A, B)

    def test_check_nodes_unexpected(self):
        A = Graph()
        B = Graph()

        A.add_nodes_from([1, 2, 3])
        B.add_nodes_from([1, 2])

        try:
            self.assertTopologyEqual(A, B)
        except AssertionError as e:
            emsg = str(e).lower()
            if 'unexpected nodes' not in emsg:
                self.fail('Unexpected error/assert raised')
        else:
            self.fail('Allowed graph mismatch')

    def test_check_nodes_missing(self):
        A = Graph()
        B = Graph()

        A.add_nodes_from([1, 2])
        B.add_nodes_from([1, 2, 3])

        try:
            self.assertTopologyEqual(A, B)
        except AssertionError as e:
            emsg = str(e).lower()
            if 'missing nodes' not in emsg:
                self.fail('Unexpected error/assert raised')
        else:
            self.fail('Allowed graph mismatch')

    def test_check_edges_unexpected(self):
        A = Graph()
        B = Graph()

        A.add_nodes_from([1, 2, 3])
        B.add_nodes_from([1, 2, 3])
        A.add_path([1, 2, 3])
        B.add_path([1, 2])

        try:
            self.assertTopologyEqual(A, B)
        except AssertionError as e:
            emsg = str(e).lower()
            if 'unexpected edges' not in emsg:
                self.fail('Unexpected error/assert raised')
        else:
            self.fail('Allowed graph mismatch')

    def test_check_edges_missing(self):
        A = Graph()
        B = Graph()

        A.add_nodes_from([1, 2, 3])
        B.add_nodes_from([1, 2, 3])
        A.add_path([1, 2])
        B.add_path([1, 2, 3])

        try:
            self.assertTopologyEqual(A, B)
        except AssertionError as e:
            emsg = str(e).lower()
            if 'missing edges' not in emsg:
                self.fail('Unexpected error/assert raised')
        else:
            self.fail('Allowed graph mismatch')

    def test_check_all_problems(self):
        A = Graph()
        B = Graph()

        A.add_path([1, 2])
        B.add_path([2, 3])

        try:
            self.assertTopologyEqual(A, B)
        except AssertionError as e:
            emsg = str(e).lower()
            if not ('missing edges' in emsg and
                    'missing nodes' in emsg and
                    'unexpected edges' in emsg and
                    'unexpected nodes' in emsg):
                self.fail('Unexpected error/assert raised')
        else:
            self.fail('Allowed graph mismatch')


class GraphCreator(GraphTestCase):
    def test_empty_init(self):
        Graph()

    def test_init_with_graph(self):
        G = nx.DiGraph()
        G.add_path([1, 2, 3])
        Graph(G)
