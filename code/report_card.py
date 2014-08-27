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
import wio.file_manager as fm

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
                durations.append(digraph.lifespan_f(node))
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
        #collider.collapse_group_of_nodes(graph, max_duration=5)  # 5 seconds
        collider.assimilate(graph, max_threshold=10)
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

def collision_iteration2(experiment, graph):

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

        ############### Cut Worms
        candidates = collider.find_potential_cut_worms(graph, experiment,
                                                       max_first_last_distance=40, max_sibling_distance=50, debug=False)
        for candidate in candidates:
            graph.condense_nodes(candidate[0], *candidate[1:])
        report_card.add_step(graph, 'cut_worms ({n})'.format(n=len(candidates)))

        ############### Collisions

        suspects = collider.find_area_based_collisions(graph, experiment)
        suspects = [node for pred1, pred2, node, succ1, succ2 in suspects]
        print('{n} suspects found with area difference'.format(n=len(suspects)))
        collider.resolve_collisions(graph, experiment, suspects)

        threshold=2
        suspects2 = collider.suspected_collisions(graph, threshold)
        print('{n} suspects found with time difference'.format(n=len(suspects2)))
        collider.resolve_collisions(graph, experiment, suspects2)

        report_card.add_step(graph, 'collisions ({n})'.format(n=len(suspects) + len(suspects2)))

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
        graph2, df = collision_iteration(experiment, graph.copy())
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
    return experiment, graph

def determine_lost_and_found_causes(experiment, graph):


    # create a basic dataframe with information about all blob terminals

    terms = experiment.prepdata.load('terminals')
    terms = terms[np.isfinite(terms['t0'])]
    if 'bid' in terms.columns:
        terms.set_index('bid', inplace=True)
    term_ids = set(terms.index) # get set of all bids with data
    terms['node_id'] = 0
    terms['id_change_found'] = False
    terms['id_change_lost'] = False

    # loop through graph and assign all blob ids to cooresponding nodes.
    # also include if nodes have parents or children

    for node_id in graph.nodes(data=False):
        successors = list(graph.successors(node_id))
        predecessors = list(graph.predecessors(node_id))
        has_pred = len(predecessors) > 0
        has_suc = len(successors) > 0
        comps = graph[node_id].get('components', [node_id])
        known_comps = set(comps) & term_ids
        for comp in comps:
            terms['node_id'].loc[list(known_comps)] = node_id
            terms['id_change_found'].loc[list(known_comps)] = has_pred
            terms['id_change_lost'].loc[list(known_comps)] = has_suc

    # split dataframe into seperate dfs concerned with starts and ends
    # standardize collumn names such that both have the same columns

    start_terms = terms[['t0', 'x0', 'y0', 'f0', 'node_id', 'id_change_found']]
    start_terms.rename(columns={'t0':'t', 'x0':'x', 'y0':'y', 'f0':'f',
                                'id_change_found': 'id_change'},
                       inplace=True)

    end_terms = terms[['tN', 'xN', 'yN', 'fN', 'node_id', 'id_change_lost']]
    end_terms.rename(columns={'tN':'t', 'xN':'x', 'yN':'y', 'fN':'f',
                              'id_change_lost': 'id_change'},
                     inplace=True)

    # precautionary drop rows with NaN as 't' (most other data will be missing)
    start_terms = start_terms[np.isfinite(start_terms['t'])]
    end_terms = end_terms[np.isfinite(end_terms['t'])]

    # drop rows that have duplicate node_ids.
    # for starts, take the first row (lowest time)

    start_terms.sort(columns='t', inplace=True)
    start_terms.drop_duplicates('node_id', take_last=False,
                                inplace=True)
    # for ends, take the last row (highest time)
    end_terms.sort(columns='t', inplace=True)
    end_terms.drop_duplicates('node_id', take_last=True,
                              inplace=True)

    # mark if nodes start or end on the edge of the image.

    edge_thresh = 80
    start_thresh = 30

    plate_size = [1728, 2352]
    xlow, xhigh = edge_thresh, plate_size[0] - edge_thresh
    ylow, yhigh = edge_thresh, plate_size[1] - edge_thresh

    def add_on_edge(df):
        df['on_edge'] = False
        df['on_edge'][df['x'] < xlow] = True
        df['on_edge'][df['y'] < ylow] = True
        df['on_edge'][ xhigh < df['x']] = True
        df['on_edge'][ yhigh < df['y']] = True

    add_on_edge(start_terms)
    add_on_edge(end_terms)

    # mark if nodes start or end outside region of interest ROI
    ex_id = experiment.id
    print ex_id
    roi = fm.ImageMarkings(ex_id=ex_id).roi()
    print roi
    x, y, r = roi['x'], roi['y'], roi['r']
    def add_out_of_roi(df):
        dists = np.sqrt((df['x'] - x)**2 + (df['y'] - y)**2)
        df['outside-roi'] = dists < r


    # mark if nodes start or end with the start/end of the recording.

    start_terms['timing'] = False
    start_terms['timing'][start_terms['t'] < start_thresh] = True
    end_terms['timing'] = False
    end_terms['timing'][end_terms['t'] >= 3599] = True

    print end_terms.head()
    return start_terms, end_terms


if __name__ == '__main__':
    #main()
    experiment, graph = main2()
    determine_lost_and_found_causes(experiment, graph)

    #ex_id = '20130318_131111'

    #experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
    #graph = experiment.graph.copy()
    #collision_iteration2(experiment, graph)
    #collision_iteration(experiment, graph)
