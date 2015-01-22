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

# def cumulative_sum(seq, start=0):
#     x = start
#     yield x
#     for element in seq:
#         x += element
#         yield x

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


class GraphCheck(unittest.TestCase):
    def check_graphs_equal(self, Gtest, Gexpect):
        # # check topologies
        nGt = set(Gtest.nodes_iter())
        nGe = set(Gexpect.nodes_iter())
        eGt = set(Gtest.edges_iter())
        eGe = set(Gexpect.edges_iter())
        self.assertTrue(nGt == nGe and eGt == eGe, 'Graph mismatch.\n\n'
                'Unexpected nodes:        {}\n'
                'Expected, missing nodes: {}\n\n'
                'Unexpected edges:        {}\n'
                'Expected, missing edges: {}'.format(
                        ', '.join(str(x) for x in nGt-nGe),
                        ', '.join(str(x) for x in nGe-nGt),
                        ', '.join(str(x) for x in eGt-eGe),
                        ', '.join(str(x) for x in eGe-eGt)))

class GraphCreator(GraphCheck):
    def test_empty_init(self):
        Graph()

    def test_init_with_graph(self):
        G = nx.DiGraph()
        G.add_path([1, 2, 3])
        Graph(G)
