import unittest
import itertools

import networkx as nx

from waldo.network import Graph

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

def diamond_graph_a():
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


class CondenserTopology(GraphCheck):
    def test_forward(self):
        Gtest = diamond_graph_a()

        Gtest.condense_nodes(3, 4, 5, 6)

        Gref = nx.DiGraph()
        Gref.add_path([1, 3, 7])
        Gref.add_path([2, 3, 8])

        self.check_graphs_equal(Gtest, Gref)

    def test_backward(self):
        Gtest = diamond_graph_a()

        Gtest.condense_nodes(6, 5, 4, 3)

        Gref = nx.DiGraph()
        Gref.add_path([1, 6, 7])
        Gref.add_path([2, 6, 8])

        self.check_graphs_equal(Gtest, Gref)

    def test_unconnected(self):
        nodes = [
            [1, 4],
            [2, 5],
            [3, 6],
        ]
        Gtest = node_generate(nodes)
        Gtest.add_path([1, 2, 3])
        Gtest.add_path([4, 5, 6])

        try:
            Gtest.condense_nodes(2, 5)
        except ValueError:
            pass
        else:
            self.fail("Allowed unconnected nodes to be merged.")

    def test_partial_connected(self):
        nodes = [
            [1, 2],
            [3],
            [4, 5],
            [6],
            [7, 8],
        ]
        Gtest = node_generate(nodes)
        Gtest.add_path([1, 3, 4, 6, 7])
        Gtest.add_path([2, 3])
        Gtest.add_path([5, 6, 8])

        # shouldn't complain.
        Gtest.condense_nodes(3, 4, 5, 6)


class CondenserEdgeInfo(GraphCheck):
    def test_tagging(self):
        Gtest = diamond_graph_a()
        Gtest.tag_edges()

        for a, b, data in Gtest.edges_iter(data=True):
            self.assertEqual(
                    data['blob_id_edges'],
                    {(a, b)}
                )

    def test_keep_original_info(self):
        Gtest = diamond_graph_a()
        Gtest.tag_edges()

        Gtest.condense_nodes(3, 4, 5, 6)

        edge_tags = {
            (1, 3): {(1, 3)},
            (2, 3): {(2, 3)},
            (3, 7): {(6, 7)},
            (3, 8): {(6, 8)},
        }

        for a, b, data in Gtest.edges_iter(data=True):
            self.assertEqual(
                    data['blob_id_edges'],
                    edge_tags[(a, b)]
                )

    def test_unclean_merge(self):
        Gtest = diamond_graph_a()
        Gtest.tag_edges()

        Gtest.condense_nodes(3, 4, 5)

        edge_tags = {
            (1, 3): {(1, 3)},
            (2, 3): {(2, 3)},
            (3, 6): {(4, 6), (5, 6)},
        }

        for a, b, data in Gtest.edges_iter(data=True):
            self.assertEqual(
                    data['blob_id_edges'],
                    edge_tags[(a, b)]
                )
