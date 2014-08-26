import os
import sys
from glob import glob
import pickle
import itertools

import pandas as pd
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import prettyplotlib as ppl
from statsmodels.distributions.empirical_distribution import ECDF

import setpath
os.environ.setdefault('WALDO_SETTINGS', 'default_settings')

from conf import settings
from wio import Experiment
import tape.taper as tp
import collider

DATA_DIR = settings.LOGISTICS['filesystem_data']

class ReportCard(object):

    def __init__(self, experiment):
        self.steps = []
        self.reports = []
        self.durations = []
        self.experiment = experiment

    def add_step(self, graph, step_name):
        report, durations = self.evaluate_graph(graph)
        report['step'] = step_name
        self.steps.append(step_name)
        self.reports.append(report)
        self.durations.append(durations)


    def evaluate_graph(self, digraph, threshold=2):
        graph = digraph.to_undirected()
        isolated_count, connected_count = 0, 0
        durations = []
        components = list(nx.connected_components(graph))
        giant_size = max([len(c) for c in components])

        for node in digraph:
            parents = list(digraph.predecessors(node))
            children = list(digraph.successors(node))
            if parents or children:
                connected_count += 1
            else:
                isolated_count += 1
            try:
                durations.append(digraph.lifespan(node))
            except KeyError:
                pass

        duration_med = np.median(durations)
        duration_std = np.std(durations)
        assert len(graph.nodes(data=False)) == graph.number_of_nodes()
        report = {'total-nodes':graph.number_of_nodes(),
                  'isolated-nodes': isolated_count,
                  'connected-nodes': connected_count,
                  'giant-component-size':giant_size,
                  'duration-med': round(duration_med, ndigits=2),
                  'duration-std': round(duration_std, ndigits=2),
                  '# components': len(components),
                  'moving-nodes': len(compound_bl_filter(self.experiment,
                        digraph, threshold))
                  }
        return report, durations

    def report(self, show=True):
        columns = ['step', 'total-nodes', 'isolated-nodes',
                   'connected-nodes', 'giant-component-size',
                   'duration-med', 'duration-std', '# components',
                   'moving-nodes']

        report_df = pd.DataFrame(self.reports, columns=columns)
        report_df.set_index('step')
        if show:
            print report_df[['step', 'total-nodes', 'isolated-nodes', 'duration-med',
            'moving-nodes']]
        return report_df

# find me a better home
def compound_bl_filter(experiment, graph, threshold):
    """
    Return node IDs from *graph* and *experiment* if they moved at least
    *threshold* standard body lengths.
    """
    cbounds = compound_bounding_box(experiment, graph)
    cbounds['bl'] = ((cbounds['x_max'] - cbounds['x_min'] +
                      cbounds['y_max'] - cbounds['y_min']) /
                      experiment.typical_bodylength)
    moved = cbounds[cbounds['bl'] >= threshold]
    return moved['bid'] if 'bid' in moved.columns else moved.index

def compound_bounding_box(experiment, graph):
    """
    Construct bounding box for all nodes (compound or otherwise) by using
    experiment prepdata and graph node components.
    """
    bounds = experiment.prepdata.bounds
    return bounds.groupby(graph.where_is).agg(
            {'x_min': min, 'x_max': max, 'y_min': min, 'y_max': max})

def create_report_card(experiment, graph):

    report_card = ReportCard(experiment)
    report_card.add_step(graph, 'raw')

    ############### Remove Known Junk

    collider.remove_nodes_outside_roi(graph, experiment)
    report_card.add_step(graph, 'roi')

    collider.remove_blank_nodes(graph, experiment)
    report_card.add_step(graph, 'blank')

    ############### Simplify
    collider.collapse_group_of_nodes(graph, max_duration=5)  # 5 seconds
    #collider.assimilate(graph, max_threshold=10)
    collider.remove_single_descendents(graph)
    collider.remove_fission_fusion(graph)
    collider.remove_fission_fusion_rel(graph, split_rel_time=0.5)
    collider.remove_offshoots(graph, threshold=20)
    collider.remove_single_descendents(graph)
    report_card.add_step(graph, 'simplify')

    ############### Cut Worms
    candidates = collider.find_potential_cut_worms(graph, experiment,
                                                   max_first_last_distance=40, max_sibling_distance=50, debug=False)
    for candidate in candidates:
        graph.condense_nodes(candidate[0], *candidate[1:])
    report_card.add_step(graph, 'cut_worms ({n})'.format(n=len(candidates)))

    ############### Collisions
    threshold=2
    #suspects = collider.suspected_collisions(graph, threshold)
    suspects = collider.find_area_based_collisions(graph, experiment)
    suspects = [node for pred1, pred2, node, succ1, succ2 in suspects]
    print('{n} suspects found with area difference'.format(n=len(suspects)))
    collider.resolve_collisions(graph, experiment, suspects)
    report_card.add_step(graph, 'collisions ({n})'.format(n=len(suspects)))

    ############### Gaps

    taper = tp.Taper(experiment=experiment, graph=graph)
    start, end = taper.find_start_and_end_nodes()
    gaps = taper.score_potential_gaps(start, end)
    taper.greedy_tape(gaps, threshold=0.001, add_edges=True)
    graph = taper._graph
    report_card.add_step(graph, 'gaps')

    report_df = report_card.report(show=True)
    return graph, report_df


def collision_iteration(experiment, graph):

    report_card = ReportCard(experiment)
    report_card.add_step(graph, 'raw')

    ############### Remove Known Junk

    collider.remove_nodes_outside_roi(graph, experiment)
    report_card.add_step(graph, 'roi')

    collider.remove_blank_nodes(graph, experiment)
    report_card.add_step(graph, 'blank')


    ############### Simplify
    for i in range(6):
        print('iteration', i+1)
        ############### Simplify
        collider.collapse_group_of_nodes(graph, max_duration=5)  # 5 seconds
        #collider.assimilate(graph, max_threshold=10)
        collider.remove_single_descendents(graph)
        collider.remove_fission_fusion(graph)
        collider.remove_fission_fusion_rel(graph, split_rel_time=0.5)
        collider.remove_offshoots(graph, threshold=20)
        collider.remove_single_descendents(graph)
        report_card.add_step(graph, 'simplify')

        ############### Cut Worms
        candidates = collider.find_potential_cut_worms(graph, experiment,
                                                       max_first_last_distance=40, max_sibling_distance=50, debug=False)
        for candidate in candidates:
            graph.condense_nodes(candidate[0], *candidate[1:])
        report_card.add_step(graph, 'cut_worms ({n})'.format(n=len(candidates)))

        ############### Collisions
        threshold=2
        suspects = collider.suspected_collisions(graph, threshold)
        #suspects = collider.find_area_based_collisions(graph, experiment)
        #suspects = [node for pred1, pred2, node, succ1, succ2 in suspects]
        print('{n} suspects found with area difference'.format(n=len(suspects)))
        collider.resolve_collisions(graph, experiment, suspects)
        report_card.add_step(graph, 'collisions ({n})'.format(n=len(suspects)))

        ############### Gaps

        taper = tp.Taper(experiment=experiment, graph=graph)
        start, end = taper.find_start_and_end_nodes()
        gaps = taper.score_potential_gaps(start, end)
        taper.greedy_tape(gaps, threshold=0.001, add_edges=True)
        graph = taper._graph
        report_card.add_step(graph, 'gaps')


    report_df = report_card.report(show=True)
    return graph, report_df

def create_reports():
    ex_ids = ['20130318_131111',
              '20130614_120518',
              '20130702_135704',
              '20130614_120518']
    for ex_id in ex_ids:
        experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
        graph = experiment.graph.copy()

        savename = '{eid}-report.csv'.format(eid=ex_id)
        graph1, df = create_report_card(experiment, graph.copy())
        df.to_csv(savename)

        savename = '{eid}-iterative-report.csv'.format(eid=ex_id)
        graph2, df = collision_itteration(experiment, graph.copy())
        df.to_csv(savename)


def main():
    ex_id = '20130318_131111'
    #ex_id = '20130614_120518'
    #ex_id = '20130702_135704' # many pics
    # ex_id = '20130614_120518'

    experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
    graph = experiment.graph.copy()


    graph1, df = create_report_card(experiment, graph.copy())

def main2():
    ex_id = '20130318_131111'

    experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
    graph = experiment.graph.copy()

    report_card = ReportCard(experiment)
    report_card.add_step(graph, 'raw')

    ############### Remove Known Junk

    collider.remove_nodes_outside_roi(graph, experiment)
    report_card.add_step(graph, 'roi')

    collider.remove_blank_nodes(graph, experiment)
    report_card.add_step(graph, 'blank')

    ############### Cut Worms
    candidates = collider.find_potential_cut_worms(graph, experiment,
                                                   max_first_last_distance=40, max_sibling_distance=50)
    for candidate in candidates:
        print(candidate)
        graph.condense_nodes(candidate[0], *candidate[1:])
    report_card.add_step(graph, 'cut_worms ({n})'.format(n=len(candidates)))

    ############### Collisions
    threshold=2
    suspects = collider.suspected_collisions(graph, threshold)
    #suspects = collider.find_area_based_collisions(graph, experiment)
    #suspects = [node for pred1, pred2, node, succ1, succ2 in suspects]
    print('{n} suspects found with area difference'.format(n=len(suspects)))
    collider.resolve_collisions(graph, experiment, suspects)
    report_card.add_step(graph, 'collisions ({n})'.format(n=len(suspects)))

    ############### Simplify
    collider.collapse_group_of_nodes(graph, max_duration=5)  # 5 seconds
    #collider.assimilate(graph, max_threshold=10)
    collider.remove_single_descendents(graph)
    collider.remove_fission_fusion(graph)
    collider.remove_fission_fusion_rel(graph, split_rel_time=0.5)
    collider.remove_offshoots(graph, threshold=20)
    collider.remove_single_descendents(graph)
    report_card.add_step(graph, 'simplify')

    ############### Gaps

    taper = tp.Taper(experiment=experiment, graph=graph)
    start, end = taper.find_start_and_end_nodes()
    gaps = taper.score_potential_gaps(start, end)
    taper.greedy_tape(gaps, threshold=0.001, add_edges=True)
    graph = taper._graph
    report_card.add_step(graph, 'gaps')

    report_df = report_card.report(show=True)

if __name__ == '__main__':
    #main()
    main2()
