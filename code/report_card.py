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
import wio.file_manager as fm
from wio.experiment import Experiment
import tape.taper as tp
import collider

DATA_DIR = settings.LOGISTICS['filesystem_data']

class ReportCard(object):

    def __init__(self):
        self.steps = []
        self.reports = []
        self.durations = []

    def add_step(self, graph, step_name):
        report, durations = self.evaluate_graph(graph)
        report['step'] = step_name
        self.steps.append(step_name)
        self.reports.append(report)
        self.durations.append(durations)


    def evaluate_graph(self, digraph):
        graph = digraph.to_undirected()
        isolated_count, connected_count = 0, 0
        durations = []
        components = nx.connected_components(graph)
        giant_size = max([len(c) for c in components])

        for node in digraph:
            parents = list(digraph.predecessors(node))
            children = list(digraph.successors(node))
            if parents or children:
                connected_count += 1
            else:
                isolated_count += 1
            try:
                durations.append(digraph.node[node]['died'] - digraph.node[node]['born'])
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
                  }
        return report, durations

    def report(self, show=True):
        columns = ['step', 'total-nodes', 'isolated-nodes',
                   'connected-nodes', 'giant-component-size',
                   'duration-med', 'duration-std', '# components']

        report_df = pd.DataFrame(self.reports, columns=columns)
        report_df.set_index('step')
        if show:
            print report_df[['step', 'total-nodes', 'isolated-nodes', 'duration-med']]
        return report_df

def create_report_card(experiment, graph):

    report_card = ReportCard()
    report_card.add_step(graph, 'raw')

    ############### Remove Known Junk

    collider.remove_nodes_outside_roi(graph, experiment)
    report_card.add_step(graph, 'roi')

    collider.remove_blank_nodes(graph, experiment)
    report_card.add_step(graph, 'blank')

    ############### Simplify

    #collider.assimilate(graph, max_threshold=10)
    collider.remove_single_descendents(graph)
    collider.remove_fission_fusion(graph)
    collider.remove_fission_fusion_rel(graph, split_rel_time=0.5)
    collider.remove_offshoots(graph, threshold=20)
    collider.remove_single_descendents(graph)
    report_card.add_step(graph, 'simplify')

    ############### Collisions
    threshold=2
    suspects = collider.suspected_collisions(graph, threshold)
    print('{n} suspects found with time difference'.format(n=len(suspects)))
    collider.resolve_collisions(graph, experiment, suspects)
    report_card.add_step(graph, 'collisions')

    ############### Gaps

    taper = tp.Taper(experiment=experiment, graph=graph)
    start, end = taper.find_start_and_end_nodes()
    gaps = taper.score_potential_gaps(start, end)
    taper.greedy_tape(gaps, threshold=0.001, add_edges=True)
    graph = taper._graph
    report_card.add_step(graph, 'gaps')

    report_df = report_card.report(show=True)
    return graph, report_df


def main():
    ex_id = '20130318_131111'
    ex_id = '20130614_120518'
    #ex_id = '20130702_135704' # many pics
    # ex_id = '20130614_120518'

    experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
    graph = experiment.graph.copy()
    return create_report_card(experiment, graph)

if __name__ == '__main__':
    main()
