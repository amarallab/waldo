from waldo.network import keyconsts as kc

from . import test_graph as tg

class EdgeInfoChecker(tg.GraphCheck):
    def check_expected_edge_info(self, Gtest, edge_tags):
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

    def null_expected_tags(self, Gtest):
        edge_tags = {}

        for edge in Gtest.edges_iter():
            edge_tags[edge] = {edge}

        return edge_tags


class CondenserEdgeInfo(EdgeInfoChecker):
    def test_tagging(self):
        Gtest = tg.diamond_graph_a(tag=True)

        for a, b, data in Gtest.edges_iter(data=True):
            try:
                edgeids = data[kc.BLOB_ID_EDGES]
            except KeyError:
                self.fail('Data for edge {} -> {} does not have key "{}"'
                        .format(a, b, kc.BLOB_ID_EDGES))

            self.assertEqual(
                    edgeids,
                    {(a, b)}
                )

    def test_keep_original_info(self):
        Gtest = tg.diamond_graph_a(tag=True)

        Gtest.condense_nodes(3, 4, 5, 6)

        edge_tags = {
            (1, 3): {(1, 3)},
            (2, 3): {(2, 3)},
            (3, 7): {(6, 7)},
            (3, 8): {(6, 8)},
        }

        self.check_expected_edge_info(Gtest, edge_tags)

    def test_tag_type_semiirrelevant(self):
        Gtest = tg.diamond_graph_a(tag=True)

        def f(n):
            return 'node {}'.format(n)

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

        self.check_expected_edge_info(Gtest, edge_tags)

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

        self.check_expected_edge_info(Gtest, edge_tags)

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

        self.check_expected_edge_info(Gtest, edge_tags)

    def test_multiple_merges_backwards(self):
        Gtest = tg.graph_b(tag=True)

        Gtest.condense_nodes(41, 31, 32, 21)
        Gtest.condense_nodes(61, 51, 52, 41)

        edge_tags = self.multimerge_tags(61)

        self.check_expected_edge_info(Gtest, edge_tags)

    def test_multiple_merges_bothways(self):
        Gtest = tg.graph_b(tag=True)

        Gtest.condense_nodes(41, 31, 32, 21)
        Gtest.condense_nodes(41, 51, 52, 61)

        edge_tags = self.multimerge_tags(41)

        self.check_expected_edge_info(Gtest, edge_tags)


class CollisionEdgeInfo(EdgeInfoChecker):
    def test_collision_resolve(self):
        Gtest = tg.diamond_graph_a(tag=True)
        Gtest.remove_nodes_from([6, 7, 8]) # an X remains.

        Gtest.untangle_collision(3, [[1, 4], [2, 5]])

        edge_tags = {
            (1, 4): {(1, 4)},
            (2, 5): {(2, 5)},
        }

        self.check_expected_edge_info(Gtest, edge_tags)

    def test_condensed_collision(self):
        Gtest = tg.diamond_graph_a(tag=True)

        Gtest.condense_nodes(3, 4, 5, 6)
        Gtest.untangle_collision(3, [[1, 7], [2, 8]])

        edge_tags = {
            (1, 7): {(1, 7)},
            (2, 8): {(2, 8)},
        }

        self.check_expected_edge_info(Gtest, edge_tags)

class GapEdgeInfo(EdgeInfoChecker):
    def test_gap_close(self):
        pass
