import networkx as nx

from waldo.network import keyconsts as kc
from waldo.network import Graph

from . import test_graph as tg

class CondenserComponentNodeInfo(tg.GraphTestCase):
    def test_components(self):
        nodes = [[1], [2]]
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([1, 2])

        Gtest.condense_nodes(1, 2)

        self.assertEqual(Gtest.node[1][kc.COMPONENTS], {1, 2})

    def test_apriori_components_main(self):
        nodes = [[1], [2]]
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([1, 2])
        Gtest.node[1][kc.COMPONENTS] = {1, 11}

        Gtest.condense_nodes(1, 2)

        self.assertEqual(Gtest.node[1][kc.COMPONENTS], {1, 11, 2})

    def test_apriori_components_sub(self):
        nodes = [[1], [2]]
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([1, 2])
        Gtest.node[2][kc.COMPONENTS] = {2, 22}

        Gtest.condense_nodes(1, 2)

        self.assertEqual(Gtest.node[1][kc.COMPONENTS], {1, 2, 22})

    def test_multi_condense_components(self):
        nodes = [[1], [2], [3]]
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([1, 2, 3])

        Gtest.condense_nodes(1, 2)
        Gtest.condense_nodes(1, 3)

        self.assertEqual(Gtest.node[1][kc.COMPONENTS], {1, 2, 3})


class CondenserEdgeInfo(tg.GraphTestCase):
    def test_tagging(self):
        nodes = [[1], [2], [3]]
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([1, 2, 3])

        Gtest.tag_edges()

        class GetitemSetLoopback:
            def __getitem__(self, key):
                return {key}

        self.assertEdgeInfo(Gtest, GetitemSetLoopback())

    def test_keep_original_info(self):
        Gtest = tg.diamond_graph_a(tag=True)

        Gtest.condense_nodes(3, 4, 5, 6)

        edge_tags = {
            (1, 3): {(1, 3)},
            (2, 3): {(2, 3)},
            (3, 7): {(6, 7)},
            (3, 8): {(6, 8)},
        }

        self.assertEdgeInfo(Gtest, edge_tags)

    def test_arbitrary_edge_tag_names(self):
        Gtest = tg.diamond_graph_a(tag=True)

        def f(n):
            return 'node {}!'.format(n)

        # rename all the tags
        for a, b, data in Gtest.edges_iter(data=True):
            edge_tags = data[kc.BLOB_ID_EDGES]
            mod_tags = {(f(x), f(y)) for x, y in edge_tags}
            data[kc.BLOB_ID_EDGES] = mod_tags

        Gtest.condense_nodes(3, 4, 5, 6)

        edge_tags = {
            (1, 3): {(f(1), f(3))},
            (2, 3): {(f(2), f(3))},
            (3, 7): {(f(6), f(7))},
            (3, 8): {(f(6), f(8))},
        }

        self.assertEdgeInfo(Gtest, edge_tags)

    def test_unclean_merge(self):
        # maybe this should fail...
        Gtest = tg.diamond_graph_a(tag=True)

        Gtest.condense_nodes(3, 4, 5)

        edge_tags = {
            (1, 3): {(1, 3)},
            (2, 3): {(2, 3)},
            (3, 6): {(4, 6), (5, 6)},
            (6, 7): {(6, 7)},
            (6, 8): {(6, 8)},
        }

        self.assertEdgeInfo(Gtest, edge_tags)

    def multimerge_tags(self, remaining_node):
        edge_tags = {
            (11, remaining_node): {(11, 21)},
            (12, remaining_node): {(12, 21)},
            (remaining_node, 71): {(61, 71)},
            (remaining_node, 72): {(61, 72)},
        }
        return edge_tags

    def test_multiple_merges(self):
        Gtest = tg.graph_b(tag=True)

        Gtest.condense_nodes(21, 31, 32, 41)
        Gtest.condense_nodes(21, 51, 52, 61)

        edge_tags = self.multimerge_tags(21)

        self.assertEdgeInfo(Gtest, edge_tags)

    def test_multiple_merges_backwards(self):
        Gtest = tg.graph_b(tag=True)

        Gtest.condense_nodes(41, 31, 32, 21)
        Gtest.condense_nodes(61, 51, 52, 41)

        edge_tags = self.multimerge_tags(61)

        self.assertEdgeInfo(Gtest, edge_tags)

    def test_multiple_merges_bothways(self):
        Gtest = tg.graph_b(tag=True)

        Gtest.condense_nodes(41, 31, 32, 21)
        Gtest.condense_nodes(41, 51, 52, 61)

        edge_tags = self.multimerge_tags(41)

        self.assertEdgeInfo(Gtest, edge_tags)


class CondenserTopology(tg.GraphTestCase):
    def test_forward(self):
        Gtest = tg.diamond_graph_a()

        Gtest.condense_nodes(3, 4, 5, 6)

        Gref = nx.DiGraph()
        Gref.add_path([1, 3, 7])
        Gref.add_path([2, 3, 8])

        self.assertTopologyEqual(Gtest, Gref)

    def test_backward(self):
        Gtest = tg.diamond_graph_a()

        Gtest.condense_nodes(6, 5, 4, 3)

        Gref = nx.DiGraph()
        Gref.add_path([1, 6, 7])
        Gref.add_path([2, 6, 8])

        self.assertTopologyEqual(Gtest, Gref)

    def test_partial_connected(self):
        nodes = [
            [1, 2],
            [3],
            [4, 5],
            [6],
            [7, 8],
        ]
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([1, 3, 4, 6, 7])
        Gtest.add_path([2, 3])
        Gtest.add_path([5, 6, 8])

        # shouldn't complain.
        Gtest.condense_nodes(3, 4, 5, 6)

    def test_unconnected_reject(self):
        nodes = [
            [1, 2],
        ]
        Gtest = tg.node_generate(nodes)

        try:
            Gtest.condense_nodes(1, 2)
        except ValueError as e:
            if 'unconnected' not in str(e):
                self.fail('Unexpected error')
        else:
            self.fail('Allowed merging of unconnected nodes')

    def test_unconnected_override(self):
        nodes = [
            [1, 2],
        ]
        Gtest = tg.node_generate(nodes)

        try:
            Gtest.condense_nodes(1, 2, enforce_connectivity=False)
        except ValueError as e:
            if 'unconnected' in str(e):
                self.fail('Ignored override to merge unconnected nodes')
            else:
                raise e # unknown error


class CondenserHeadTailNodeInfo(tg.GraphTestCase):
    def test_unmodified(self):
        Gtest = tg.node_generate([[1]])
        Gtest.tag_edges()
