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
                dur = digraph.lifespan_t(node)
                if not np.isnan(dur):
                    durations.append(dur)
            except KeyError:
                pass

        duration_med = np.median(durations)
        duration_std = np.std(durations)
        n_nodes = graph.number_of_nodes()
        n_10_min = len([d for d in durations if d >= 600])
        n_20_min = len([d for d in durations if d >= 1200])
        n_30_min = len([d for d in durations if d >= 1200])
        # if len(durations) < n_nodes:
        #     print '{x} durations found for {y} nodes'.format(x=len(durations),
        #                                                  y=n_nodes)
        #     print duration_med, duration_std
        #     print round(duration_med, ndigits=2), round(duration_std, ndigits=2)

        assert len(graph.nodes(data=False)) == n_nodes
        report = {'total-nodes':graph.number_of_nodes(),
                  'isolated-nodes': isolated_count,
                  'connected-nodes': connected_count,
                  'giant-component-size':giant_size,
                  'duration-med': round(duration_med, ndigits=2),
                  'duration-std': round(duration_std, ndigits=2),
                  '# components': len(components),
                  '10min':  n_10_min,
                  '20min':  n_20_min,
                  '30min':  n_30_min,
                  'moving-nodes': len(compound_bl_filter(self.experiment,
                        digraph, threshold))
                  }
        return report, durations

    def report(self, show=True):
        columns = ['step', 'total-nodes', '10min', '20min', '30min',
                   'isolated-nodes', 'connected-nodes', 'giant-component-size',
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
    n = collision_suite(experiment, graph)
    report_card.add_step(graph, 'collisions ({n})'.format(n=n))

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
        n = collision_suite(experiment, graph)
        report_card.add_step(graph, 'collisions ({n})'.format(n=n))

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
        n = collision_suite(experiment, graph)
        report_card.add_step(graph, 'collisions ({n})'.format(n=n))


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


def main2(ex_id = '20130318_131111'):

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
        graph.condense_nodes(candidate[0], *candidate[1:])
    report_card.add_step(graph, 'cut_worms ({n})'.format(n=len(candidates)))

    ############### Collisions
    n = collision_suite(experiment, graph)
    report_card.add_step(graph, 'collisions ({n})'.format(n=n))

    ############### Gaps

    taper = tp.Taper(experiment=experiment, graph=graph)
    start, end = taper.find_start_and_end_nodes()
    gaps = taper.score_potential_gaps(start, end)
    taper.greedy_tape(gaps, threshold=0.001, add_edges=True)
    graph = taper._graph
    report_card.add_step(graph, 'gaps')

    ############### Simplify
    collider.collapse_group_of_nodes(graph, max_duration=5)  # 5 seconds
    #collider.assimilate(graph, max_threshold=10)
    collider.remove_single_descendents(graph)
    collider.remove_fission_fusion(graph)
    collider.remove_fission_fusion_rel(graph, split_rel_time=0.5)
    collider.remove_offshoots(graph, threshold=20)
    collider.remove_single_descendents(graph)
    report_card.add_step(graph, 'simplify')

    report_df = report_card.report(show=True)
    return experiment, graph, report_df

def collision_suite(experiment, graph, verbose=True):

    # bounding box method
    print 'collisions from bbox'

    # initialize records
    resolved = set()
    overlap_fails = set()
    data_fails = set()
    dont_bother = set()

    #trying_new_suspects = True
    collisions_were_resolved = True
    while collisions_were_resolved:
        suspects = set(collider.find_bbox_based_collisions(graph, experiment))
        s = suspects - dont_bother
        ty = len(s & data_fails)
        if verbose:
            print('\t{s} suspects. trying {ty} again'.format(s=len(s),
                                                             ty=ty))

        if not s:
            collisions_were_resolved = False
            break

        report = collider.resolve_collisions(graph, experiment,
                                             list(s))

        newly_resolved = set(report['resolved'])
        resolved = resolved | newly_resolved
        collisions_were_resolved = len(newly_resolved) > 0

        # keep track of all fails that had missing data that have
        # not been resolved yet
        data_fails = data_fails | set(report['missing_data'])
        #data_fails = data_fails - resolved

        # if overlap fails but data is missing try again later
        # if overlap fails but data is there, dont bother
        overlap_fails = overlap_fails | set(report['no_overlap'])
        overlap_fails = overlap_fails - resolved
        dont_bother = overlap_fails - data_fails

    if verbose:
        full_set = resolved | overlap_fails | data_fails | dont_bother
        full_count = len(full_set)

        n_res = len(resolved)
        n_dat = len(data_fails)
        no1 = len(data_fails & overlap_fails)
        no2 = len(dont_bother)

        p_res = int(100.0 * n_res/float(full_count))
        p_dat = int(100.0 * len(data_fails)/float(full_count))
        p_no1 = int(100.0 * no1/float(full_count))
        p_no2 = int(100.0 * no2/float(full_count))

        print '\t{n} resolved {p}%'.format(n=n_res, p=p_res)
        print '\t{n} missing data {p}%'.format(n= n_dat, p=p_dat)
        print '\t{n} missing data, no overlap {p}%'.format(n=no1, p=p_no1)
        print '\t{n} full data no,  overlap {p}%'.format(n=no2, p=p_no2)

        print '\ttrying to unzip collisions'
        collision_nodes = list(dont_bother)
        unzip_resolve_collisions(graph, experiment, collision_nodes,
                                 verbose=False, yield_bevahior=False)

    #print 'collisions from time'
    # new_suspects = set(collider.find_area_based_collisions(graph, experiment))
    # suspects = list(new_suspects - tried_suspects)
    # tried_suspects = new_suspects | tried_suspects
    # n_area = collider.resolve_collisions(graph, experiment, suspects)
    # n += n_area

    # print(int(float(n) * 100. / float(len(tried_suspects))),
    #       'percent of found collisions resolved')


    # print n_area, 'collisions from area', len(new_suspects)
    # print(int(float(n) * 100. / float(len(tried_suspects))),
    #       'percent of found collisions resolved')

    # New_suspects = set(collider.find_time_based_collisions(graph, 10, 2))
    # suspects = list(new_suspects - tried_suspects)
    # tried_suspects = new_suspects | tried_suspects
    # n_time = collider.resolve_collisions(graph, experiment, suspects)
    # n += n_time
    return len(resolved)

def determine_lost_and_found_causes(experiment, graph):

    # create a basic dataframe with information about all blob terminals
    terms = experiment.prepdata.load('terminals')
    terms = terms[np.isfinite(terms['t0'])]
    if 'bid' in terms.columns:
        terms.set_index('bid', inplace=True)
    term_ids = set(terms.index) # get set of all bids with data
    terms['node_id'] = 0
    terms['n-blobs'] = 1
    terms['id_change_found'] = False
    terms['id_change_lost'] = False
    terms['lifespan_t'] = -1
    # loop through graph and assign all blob ids to cooresponding nodes.
    # also include if nodes have parents or children

    for node_id in graph.nodes(data=False):
        successors = list(graph.successors(node_id))
        predecessors = list(graph.predecessors(node_id))
        has_pred = len(predecessors) > 0
        has_suc = len(successors) > 0
        node_data = graph.node[node_id]
        comps = node_data.get('components', [node_id])
        known_comps = set(comps) & term_ids
        for comp in comps:
            terms['n-blobs'].loc[list(known_comps)] = len(comps)
            terms['node_id'].loc[list(known_comps)] = node_id
            terms['id_change_found'].loc[list(known_comps)] = has_pred
            terms['id_change_lost'].loc[list(known_comps)] = has_suc
            terms['lifespan_t'].loc[list(known_comps)] = graph.lifespan_t(node_id)
    node_set = set(graph.nodes(data=False))
    print len(term_ids), 'blobs have terminal data'
    print len(node_set), 'nodes in graph'
    print len(term_ids & node_set), 'overlap'
    compound_nodes = set(terms[terms['n-blobs'] > 1]['node_id'])
    print len(compound_nodes), 'have more than 1 blob id in them'

    # split dataframe into seperate dfs concerned with starts and ends
    # standardize collumn names such that both have the same columns

    start_terms = terms[['t0', 'x0', 'y0', 'f0', 'node_id', 'id_change_found', 'lifespan_t']]
    start_terms.rename(columns={'t0':'t', 'x0':'x', 'y0':'y', 'f0':'f',
                                'id_change_found': 'id_change'},
                       inplace=True)

    end_terms = terms[['tN', 'xN', 'yN', 'fN', 'node_id', 'id_change_lost', 'lifespan_t']]
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
    #print ex_id
    roi = fm.ImageMarkings(ex_id=ex_id).roi()
    #print roi
    x, y, r = roi['x'], roi['y'], roi['r']
    def add_out_of_roi(df):
        dists = np.sqrt((df['x'] - x)**2 + (df['y'] - y)**2)
        df['outside-roi'] = dists > r

    add_out_of_roi(start_terms)
    add_out_of_roi(end_terms)

    # mark if nodes start or end with the start/end of the recording.

    start_terms['timing'] = False
    start_terms['timing'][start_terms['t'] < start_thresh] = True
    end_terms['timing'] = False
    end_terms['timing'][end_terms['t'] >= 3599] = True

    # by valueing certain reasons over others, we deterime a final reason.

    def determine_reason(df):
        df['reason'] = 'unknown'
        reasons = ['unknown', 'on_edge', 'id_change', 'outside-roi', 'timing']
        for reason in reasons[1:]:
            df['reason'][df[reason]] = reason

    determine_reason(start_terms)
    determine_reason(end_terms)

    start_terms.sort('lifespan_t', inplace=True, ascending=False)
    end_terms.sort('lifespan_t', inplace=True, ascending=False)
    return start_terms, end_terms

def summarize_loss_report(df):
    df = df.copy()
    df['lifespan_t'] = df['lifespan_t'] / 60.0
    bin_dividers = [1, 5, 10, 20, 61]
    reasons = ['unknown', 'on_edge', 'id_change', 'outside-roi', 'timing']
    #stuff = pd.DataFrame(columns=reasons, index=bin_dividers)
    data = []
    for bd in bin_dividers:
        b = df[df['lifespan_t'] < bd]
        df = df[df['lifespan_t'] >= bd]
        #print bd
        #print b.head()
        counts = {}
        for reason in reasons:
            counts[reason] = len(b[b['reason'] == reason])

        data.append(counts)

    report_summary = pd.DataFrame(data)
    report_summary['lifespan'] = bin_dividers
    report_summary.set_index('lifespan', inplace=True)
    report_summary = report_summary[['unknown', 'id_change', 'timing', 'on_edge', 'outside-roi']]
    print report_summary
    return report_summary

def create_reports():
    ex_ids = ['20130318_131111',
              '20130614_120518',
              '20130702_135704',
              '20130614_120518']


    for ex_id in ex_ids[1:]:
        experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
        graph = experiment.graph.copy()

        savename = '{eid}-report.csv'.format(eid=ex_id)
        #graph1, df = create_report_card(experiment, graph.copy())
        experiment, graph, report_df = main2(ex_id)
        report_df.to_csv(savename)

        starts, ends = determine_lost_and_found_causes(experiment, graph)

        starts_name = '{eid}-starts.csv'.format(eid=ex_id)
        df = summarize_loss_report(starts)
        df.to_csv(starts_name)

        ends_name = '{eid}-ends.csv'.format(eid=ex_id)
        df = summarize_loss_report(ends)
        df.to_csv(ends_name)

if __name__ == '__main__':
    create_reports()
    #experiment, graph, graph1 = main()
    #experiment, graph = main2()
    #determine_lost_and_found_causes(experiment, graph)

    #experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
    #graph = experiment.graph.copy()
    #collision_iteration2(experiment, graph)
    #collision_iteration(experiment, graph)
