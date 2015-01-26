import networkx as nx

from waldo.network import keyconsts as kc

from . import test_graph as tg

def single_untangle():
    d = {
        'G': tg.diamond_graph_a(tag=True),

        # collision node
        'x': 3,
        # parents
        'p1': 1,
        'p2': 2,
        # children
        'c1': 4,
        'c2': 5,
    }

    d['G'].untangle_collision(d['x'], [[d['p1'], d['c1']], [d['p2'], d['c2']]])

    return d

def double_untangle():
    d = {
        'G': tg.diamond_graph_a(tag=True),

        # collision node A
        'xa': 3,
        # parents
        'pa1': 1,
        'pa2': 2,
        # children
        'ca1': 4,
        'ca2': 5,

        # collision node B
        'xb': 6,
        # parents
        'pb1': 1,
        'pb2': 2,
        # children
        'cb1': 7,
        'cb2': 8,
    }

    d['G'].untangle_collision(d['xa'], [[d['pa1'], d['ca1']], [d['pa2'], d['ca2']]])
    d['G'].untangle_collision(d['xb'], [[d['pb1'], d['cb1']], [d['pb2'], d['cb2']]])

    return d


class CollisionNodeInfo(tg.GraphTestCase):
    def test_collision_tagging(self):
        d = single_untangle()

        try:
            self.assertEqual(d['G'].node[d['p1']][kc.COLLISIONS], {d['x']})
            self.assertEqual(d['G'].node[d['p2']][kc.COLLISIONS], {d['x']})
        except KeyError as e:
            if str(e) == repr(kc.COLLISIONS):
                self.fail('Node data missing collision information')
            else:
                raise e # something probably wrong with topology

    def test_collision_components(self):
        d = single_untangle()

        try:
            self.assertEqual(d['G'].node[d['p1']][kc.COMPONENTS], {d['p1'], d['c1']})
            self.assertEqual(d['G'].node[d['p2']][kc.COMPONENTS], {d['p2'], d['c2']})
        except KeyError as e:
            if str(e) == repr(kc.COLLISIONS):
                self.fail('Node data missing components')
            else:
                raise e # something probably wrong with topology.

    def test_multi_collision_tagging(self):
        d = double_untangle()

        try:
            self.assertEqual(d['G'].node[d['pa1']][kc.COLLISIONS], {d['xa'], d['xb']})
            self.assertEqual(d['G'].node[d['pa2']][kc.COLLISIONS], {d['xa'], d['xb']})
        except KeyError as e:
            if str(e) == repr(kc.COLLISIONS):
                self.fail('Node data missing collision information')
            else:
                raise e # something probably wrong with topology.

    def test_multi_collision_components(self):
        d = double_untangle()

        try:
            self.assertEqual(d['G'].node[d['pa1']][kc.COMPONENTS], {d['pa1'], d['ca1'], d['cb1']})
            self.assertEqual(d['G'].node[d['pa2']][kc.COMPONENTS], {d['pa2'], d['ca2'], d['cb2']})
        except KeyError as e:
            if str(e) == repr(kc.COLLISIONS):
                self.fail('Node data missing components')
            else:
                raise e # something probably wrong with topology.


class CollisionEdgeInfo(tg.GraphTestCase):
    def test_collision_resolve(self):
        Gtest = tg.diamond_graph_a(tag=True)
        Gtest.remove_nodes_from([6, 7, 8]) # an X remains.

        Gtest.untangle_collision(3, [[1, 4], [2, 5]])

        edge_tags = {
            (1, 4): {(1, 4)},
            (2, 5): {(2, 5)},
        }

        self.assertEdgeInfo(Gtest, edge_tags)

    def test_condensed_collision(self):
        Gtest = tg.diamond_graph_a(tag=True)

        Gtest.condense_nodes(3, 4, 5, 6)
        Gtest.untangle_collision(3, [[1, 7], [2, 8]])

        edge_tags = {
            (1, 7): {(1, 7)},
            (2, 8): {(2, 8)},
        }

        self.assertEdgeInfo(Gtest, edge_tags)


class CollisionTopology(tg.GraphTestCase):
    def test_collision(self):
        d = single_untangle()

        Gref = nx.DiGraph()
        Gref.add_path([1, 6, 7])
        Gref.add_path([2, 6, 8])

        self.assertTopologyEqual(d['G'], Gref)

    def test_multi_collision(self):
        d = double_untangle()

        Gref = nx.DiGraph()
        Gref.add_nodes_from([1, 2])

        self.assertTopologyEqual(d['G'], Gref)
