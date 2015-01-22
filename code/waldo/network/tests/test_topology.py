import networkx as nx

from . import test_graph as tg

class CondenserTopology(tg.GraphCheck):
    def test_forward(self):
        Gtest = tg.diamond_graph_a()

        Gtest.condense_nodes(3, 4, 5, 6)

        Gref = nx.DiGraph()
        Gref.add_path([1, 3, 7])
        Gref.add_path([2, 3, 8])

        self.check_graphs_equal(Gtest, Gref)

    def test_backward(self):
        Gtest = tg.diamond_graph_a()

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
        Gtest = tg.node_generate(nodes)
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
        Gtest = tg.node_generate(nodes)
        Gtest.add_path([1, 3, 4, 6, 7])
        Gtest.add_path([2, 3])
        Gtest.add_path([5, 6, 8])

        # shouldn't complain.
        Gtest.condense_nodes(3, 4, 5, 6)
