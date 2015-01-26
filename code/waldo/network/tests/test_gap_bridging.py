import unittest

import networkx as nx

from waldo.network import keyconsts as kc

from . import test_graph as tg

class GapNodeInfo(tg.GraphTestCase):
    def test_basic_gap_data(self):
        nodes = [[1], [2]]
        Gtest = tg.node_generate(nodes)

        Gtest.bridge_gap(1, 2)

        self.assertNodeGapInfo(Gtest, 1, {(1, 2)})

    def test_basic_components(self):
        nodes = [[1], [2]]
        Gtest = tg.node_generate(nodes)

        Gtest.bridge_gap(1, 2)

        self.assertNodeDataEquals(kc.COMPONENTS, Gtest, 1, {1, 2})


class GapEdgeInfo(tg.GraphTestCase):
    def test_gap_close(self):
        nodes = [[1], [2], [3], [4]]
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([1, 2])
        Gtest.add_path([3, 4])
        Gtest.tag_edges()

        Gtest.bridge_gap(2, 3)

        edge_tags = {
            (1, 2): {(1, 2)},
            (2, 4): {(3, 4)},
        }

        for a, b, data in Gtest.edges_iter(data=True):
            print('{}-{}, {}'.format(a, b, data))
        self.assertEdgeInfo(Gtest, edge_tags)


class GapTopology(tg.GraphTestCase):
    def test_gap_close(self):
        nodes = [[1], [2], [3], [4]]
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([1, 2])
        Gtest.add_path([3, 4])

        Gtest.bridge_gap(2, 3)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([1, 2, 4])

        self.assertTopologyEqual(Gtest, Gexpect)

    @unittest.skip
    def test_refuse_parent_with_other_children(self):
        nodes = [[11], [21, 22]]
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([11, 21])

        try:
            Gtest.bridge_gap(11, 22)
        except ValueError as e:
            emsg = str(e).lower()
            if 'parent node' not in emsg:
                self.fail('Unexpected ValueError: {}'.format(emsg))
        else:
            self.fail('Allowed gap fusion involving parent that has other children')

    @unittest.skip
    def test_refuse_child_with_other_parents(self):
        nodes = [[11, 12], [21]]
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([11, 21])

        try:
            Gtest.bridge_gap(12, 21)
        except ValueError as e:
            emsg = str(e).lower()
            if 'child node' not in emsg:
                self.fail('Unexpected ValueError: {}'.format(emsg))
        else:
            self.fail('Allowed gap fusion involving child that has other parents')
