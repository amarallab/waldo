from __future__ import absolute_import, print_function

# standard library
import logging

# third party
import pandas as pd
import numpy as np
import networkx as nx

# package specific
import waldo.viz.subgraph as subgraph
from waldo import collider
from waldo.conf import settings
import waldo.tape.taper as tp
import waldo.wio.file_manager as fm

L = logging.getLogger(__name__)


class ReportCard(object):
    def __init__(self, experiment):
        # self.steps = []
        self.reports = []
        self.durations = []
        self.experiment = experiment
        print('HELTENA: Reportcard init {}'.format(len(self.reports)))

    def add_step(self, graph, step_name, phase_name):
        report, durations = self.evaluate_graph(graph)
        report['phase'] = phase_name
        report['step'] = step_name
        # self.steps.append(step_name)
        self.reports.append(report)
        self.durations.append(durations)
        print('HELTENA: Reportcard add_step {}'.format(len(self.reports)))

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
        duration_mean = np.mean(durations)
        duration_std = np.std(durations)
        n_nodes = graph.number_of_nodes()
        duration_min = np.array(durations) / 60.0
        n_10_min = len([d for d in duration_min if d >= 10])
        n_20_min = len([d for d in duration_min if d >= 20])
        n_30_min = len([d for d in duration_min if d >= 30])
        n_40_min = len([d for d in duration_min if d >= 40])
        n_50_min = len([d for d in duration_min if d >= 50])

        bins = [50, 40, 30, 20, 10, 0]
        worm_minutes = {}  # '10min':0, '20min':0, '30min':0, '40min':0, '50min':0}
        for b in bins:
            worm_minutes[b] = 0

        for d in duration_min:
            for b in bins:
                if d >= b:
                    worm_minutes[b] = d + worm_minutes[b]
                    break

        wm = {}
        for b in worm_minutes:
            worm_m = worm_minutes[b]
            label = 'wm_{b}min'.format(b=b)
            wm[label] = worm_m

        moving_nodes = list(digraph.compound_bl_filter(self.experiment,
                                                       threshold))

        n1 = graph.number_of_nodes()
        n2 = digraph.number_of_nodes()
        m1 = len(moving_nodes)
        m2 = len(set(moving_nodes))

        if n1 != n2:
            print('WARNING: graphs have unequan number of nodes')
            print(n1, 'nodes in graph')
            print(n2, 'nodes in digraph')
        if m1 > n1 or m1 > n2:
            print('WARNING: more moving nodes than nodes in graph')
            print(m1, 'moving nodes')
            print(m2, 'moving nodes, no repeats')

        # if len(durations) < n_nodes:
        #     print('{x} durations found for {y} nodes'.format(x=len(durations),
        #                                                  y=n_nodes))
        #     print(duration_med, duration_std)
        #     print(round(duration_med, ndigits=2), round(duration_std, ndigits=2))

        assert len(graph.nodes(data=False)) == n_nodes
        report = {'total-nodes': graph.number_of_nodes(),
                  'isolated-nodes': isolated_count,
                  'connected-nodes': connected_count,
                  'giant-component-size': giant_size,
                  'duration-mean': round(duration_mean, ndigits=2),
                  'duration-med': round(duration_med, ndigits=2),
                  'duration-std': round(duration_std, ndigits=2),
                  '# components': len(components),
                  '>10min': n_10_min,
                  '>20min': n_20_min,
                  '>30min': n_30_min,
                  '>40min': n_40_min,
                  '>50min': n_50_min,
                  'moving-nodes': len(moving_nodes)
                  }
        report.update(wm)
        return report, durations

    def report(self, show=False):
        # columns = ['step', 'total-nodes', '>10min', '>20min', '30min', '40min', '50min',
        #            'isolated-nodes', 'connected-nodes', 'giant-component-size',
        #            'duration-mean', 'duration-med', 'duration-std', '# components',
        #            'moving-nodes']

        report_df = pd.DataFrame(self.reports)
        report_df.set_index('step')
        if show:
            print(report_df[['step', 'total-nodes', 'isolated-nodes', 'duration-med',
                             'moving-nodes']])
            print(report_df[['step', 'total-nodes', '>10min', '>20min', '>30min', '>40min', '>50min']])
            print(report_df[['step', 'wm_0min', 'wm_10min', 'wm_20min', 'wm_30min', 'wm_40min', 'wm_50min']])

        print('HELTENA: Reportcard report called: {}'.format(len(self.reports)))
        return report_df

    def determine_lost_and_found_causes(self, graph):
        experiment = self.experiment

        # create a basic dataframe with information about all blob terminals
        terms = experiment.prepdata.load('terminals')
        terms = terms[np.isfinite(terms['t0'])]
        if 'bid' in terms.columns:
            terms.set_index('bid', inplace=True)
        term_ids = set(terms.index)  # get set of all bids with data
        terms['node_id'] = 0
        terms['n-blobs'] = 1
        terms['id_change_found'] = False
        terms['id_change_lost'] = False
        terms['join_found'] = False
        terms['join_lost'] = False
        terms['split_found'] = False
        terms['split_lost'] = False
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

                terms['split_found'].loc[list(known_comps)] = len(predecessors) == 1
                terms['split_lost'].loc[list(known_comps)] = len(successors) > 1

                terms['join_found'].loc[list(known_comps)] = len(predecessors) > 1
                terms['join_lost'].loc[list(known_comps)] = len(successors) == 1

                terms['lifespan_t'].loc[list(known_comps)] = graph.lifespan_t(node_id)
        node_set = set(graph.nodes(data=False))
        print(len(term_ids), 'blobs have terminal data')
        print(len(node_set), 'nodes in graph')
        print(len(term_ids & node_set), 'overlap')
        compound_nodes = set(terms[terms['n-blobs'] > 1]['node_id'])
        print(len(compound_nodes), 'have more than 1 blob id in them')

        # split dataframe into seperate dfs concerned with starts and ends
        # standardize collumn names such that both have the same columns

        start_terms = terms[['t0', 'x0', 'y0', 'f0', 'node_id', 'id_change_found',
                             'split_found', 'join_found', 'lifespan_t']]
        start_terms.rename(columns={'t0': 't', 'x0': 'x', 'y0': 'y', 'f0': 'f',
                                    'split_found': 'split',
                                    'join_found': 'join',
                                    'id_change_found': 'id_change'},
                           inplace=True)

        end_terms = terms[['tN', 'xN', 'yN', 'fN', 'node_id', 'id_change_lost',
                           'split_lost', 'join_lost', 'lifespan_t']]

        end_terms.rename(columns={'tN': 't', 'xN': 'x', 'yN': 'y', 'fN': 'f',
                                  'split_lost': 'split',
                                  'join_lost': 'join',
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
            df['on_edge'][xhigh < df['x']] = True
            df['on_edge'][yhigh < df['y']] = True

        add_on_edge(start_terms)
        add_on_edge(end_terms)

        # mark if nodes start or end outside region of interest ROI
        ex_id = experiment.id
        # print(ex_id)
        roi = fm.ImageMarkings(ex_id=ex_id).roi()
        # print(roi)
        x, y, r = roi['x'], roi['y'], roi['r']

        def add_out_of_roi(df):
            dists = np.sqrt((df['x'] - x) ** 2 + (df['y'] - y) ** 2)
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
            # reasons = ['unknown', 'on_edge', 'id_change', 'outside-roi', 'timing']
            reasons = ['unknown', 'on_edge', 'split', 'join', 'outside-roi', 'timing']
            for reason in reasons[1:]:
                df['reason'][df[reason]] = reason

        determine_reason(start_terms)
        determine_reason(end_terms)

        start_terms.sort('lifespan_t', inplace=True, ascending=False)
        end_terms.sort('lifespan_t', inplace=True, ascending=False)
        return start_terms, end_terms

    def summarize_loss_report(self, df):
        df_orig = df
        df = df.copy()
        df['lifespan_t'] = df['lifespan_t'] / 60.0
        bin_dividers = [1, 5, 10, 20, 61]
        # reasons = ['unknown', 'on_edge', 'id_change', 'outside-roi', 'timing']
        reasons = ['unknown', 'on_edge', 'split', 'join', 'outside-roi', 'timing']
        # stuff = pd.DataFrame(columns=reasons, index=bin_dividers)
        data = []
        for bd in bin_dividers:
            b = df[df['lifespan_t'] < bd]
            df = df[df['lifespan_t'] >= bd]
            # print(bd)
            # print(b.head())
            counts = {}
            for reason in reasons:
                counts[reason] = len(b[b['reason'] == reason])

            data.append(counts)

        counts = {}
        for reason in reasons:
            counts[reason] = len(df_orig[df_orig['reason'] == reason])
        data.append(counts)

        report_summary = pd.DataFrame(data)
        report_summary['lifespan'] = bin_dividers + ['total']
        report_summary.set_index('lifespan', inplace=True)
        # report_summary = report_summary[['unknown', 'id_change', 'timing', 'on_edge', 'outside-roi']]
        report_summary = report_summary[['unknown', 'split', 'join', 'timing', 'on_edge', 'outside-roi']]
        print(report_summary)
        return report_summary

    def save_reports(self, graph):
        experiment = self.experiment
        node_info = graph.node_summary(experiment)
        experiment.prepdata.dump('node-summary', node_info)
        report = self.report()
        experiment.prepdata.dump('report-card', report)
        starts, ends = self.determine_lost_and_found_causes(graph)
        experiment.prepdata.dump('starts', starts)
        experiment.prepdata.dump('ends', ends)
        start_report = self.summarize_loss_report(starts)
        end_report = self.summarize_loss_report(ends)
        experiment.prepdata.dump('start_report', start_report)
        experiment.prepdata.dump('end_report', end_report)
        return report


class SubgraphRecorder(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.subgraphs = []

    def save_subgraph(self, graph):
        new_subgraph = self.extract_subgraph(graph)
        self.subgraphs.append(new_subgraph)

    def extract_subgraph(self, graph):
        """
        """

        def relevant_subgraph(digraph, nodes):
            subgraph = digraph.copy()
            unwanted_nodes = [n for n in subgraph.nodes(data=False) if n not in nodes]
            subgraph.remove_nodes_from(unwanted_nodes)
            return subgraph

        check_set = set(self.nodes)
        graph_nodes = set(graph.nodes(data=False))
        check_set = check_set & graph_nodes
        all_relevant_nodes = set([])
        # print(check_set)
        # print(len(check_set))
        while len(check_set):
            current_node = check_set.pop()
            subgraph = subgraph.nearby(digraph=graph, target=current_node, max_distance=10000)
            # print(subgraph)
            subgraph_nodes = set(subgraph.nodes(data=False))
            check_set = check_set - subgraph_nodes
            all_relevant_nodes = all_relevant_nodes | subgraph_nodes
        print(len(all_relevant_nodes), 'nodes in subgraph')
        return relevant_subgraph(digraph=graph, nodes=list(all_relevant_nodes))


class WaldoSolver(object):
    def __init__(self, experiment, graph):
        self.graph = graph
        self.experiment = experiment
        self.ex_id = experiment.id
        self.report_card = ReportCard(experiment)
        self.taper = tp.Taper(experiment=experiment, graph=graph)
        self.report_card.add_step(graph, step_name='raw', phase_name='input')
        self.phase_name = 'input'

    def run(self, callback=None, redraw_callback=None):
        """ runs full solver code
        """
        if callback:
            self.initial_clean(callback=lambda x: callback(x * 0.2))
            self.solve(callback=lambda x: callback(0.2 + x * 0.6), redraw_callback=redraw_callback)
            self.write_reports()
            return self.report()
        else:
            self.initial_clean()
            self.solve(redraw_callback=redraw_callback)
            self.write_reports()
            return self.report()

    def initial_clean(self, callback=None):
        """ removes blobs that are outside of the region of interest
        """
        # graph = self.graph
        experiment = self.experiment
        collider.remove_nodes_outside_roi(self.graph, experiment)
        phase_name = 'pre-cleaning'
        self.phase_name = phase_name
        if callback:
            callback(0.25)
        self.report_card.add_step(self.graph, step_name='roi',
                                  phase_name=phase_name)
        if callback:
            callback(0.5)
        collider.remove_blank_nodes(self.graph, experiment)
        if callback:
            callback(0.75)
        self.report_card.add_step(self.graph, step_name='blank',
                                  phase_name=phase_name)
        if callback:
            callback(1)

    def prune(self, threshold=20):
        """ removes leaf nodes that last less than a certain number of frames

        params
        -----
        threshold: (int)
            leaf nodes that last less than threshold frames are removed from graph
        """
        L.warn('Remove Offshoots')
        collider.remove_offshoots(self.graph, threshold=threshold)
        self.report_card.add_step(self.graph, step_name='prune',
                                  phase_name=self.phase_name)

    def consolidate(self, max_duration=None, split_rel_time=None,
                    fission_fusion_max=None):
        """ consolidates motifs that involve false splits and single
        straight lines.

        params
        -----
        max_duration: (int)
        split_rel_time: (float)

        """
        if max_duration is None:
            max_duration = settings.COLLIDER_SUITE_ASSIMILATE_SIZE
        if split_rel_time is None:
            split_rel_time = settings.COLLIDER_SUITE_SPLIT_REL
        if fission_fusion_max is None:
            fission_fusion_max = settings.COLLIDER_SUITE_SPLIT_ABS

        L.warn('Collapse Group')
        print('collapse g')
        collider.collapse_group_of_nodes(self.graph, max_duration=max_duration)
        # self.graph.deep_validate(experiment=self.experiment)

        print('fission-fusion')
        L.warn('Remove Fission-Fusion')
        collider.remove_fission_fusion(self.graph, max_split_frames=fission_fusion_max)
        # self.graph.deep_validate(experiment=self.experiment)

        print('fission-fusion rel')
        L.warn('Remove Fission-Fusion (relative)')
        collider.remove_fission_fusion_rel(self.graph, split_rel_time=split_rel_time)
        # self.graph.deep_validate(experiment=self.experiment)

        print('remove single descendents')
        L.warn('Remove Single Descendents')
        collider.remove_single_descendents(self.graph)
        # self.graph.deep_validate(experiment=self.experiment)
        self.report_card.add_step(self.graph, step_name='consolidate',
                                  phase_name=self.phase_name)

    def connect_leaves(self, gap_validation=None):
        """ draws arcs between unconected leaf nodes
        """
        L.warn('gaps')
        gap_start, gap_end = self.taper.find_start_and_end_nodes(use_missing_objects=True)
        gaps = self.taper.score_potential_gaps(gap_start, gap_end)
        if gap_validation is not None:
            gap_validation.append(gaps[['blob1', 'blob2']])

        # Score is based on (delta t) * (delta dist)
        ll1, gaps = self.taper.short_tape(gaps, add_edges=True)
        # Score is based on probability that a blob would move a certain distance (from other worms on plate)
        # ll2, gaps = self.taper.greedy_tape(gaps, threshold=0.001, add_edges=True)
        # link_total = len(ll1) #+ len(ll2) + len(ll3)
        self.report_card.add_step(self.graph, step_name='infer gaps',
                                  phase_name=self.phase_name)

    def solve(self, iterations=6, validate_steps=True, subgraph_recorder=None, callback=None, redraw_callback=None):
        """iterativly loop through (1) untangleing collisions (2)
        pruning (3) condensing and (4) infer gaps

        """
        last_graph = None
        # gap_validation = []
        # graph = self.graph

        def boiler_plate(validate_steps, subgraph_recorder):
            if validate_steps:
                self.graph.validate()
                # self.graph.deep_validate(experiment=self.experiment)
            # L.warn('Iteration {}'.format(i + 1))
            if subgraph_recorder is not None:
                subgraph_recorder.save_subgraph(self.graph)

        if callback:
            def cb_iterate(i, x):
                callback(i / 6.0 + x / 6.0)
        else:
            def cb_iterate(i, x):
                pass

        # self.report_card.add_step(self.graph, 'iter 0')
        for i in range(6):
            self.phase_name = 'iter{i}'.format(i=i + 1)
            # untangle collisions
            n = self.untangle_collsions()
            print('--- collisions ---')
            boiler_plate(validate_steps, subgraph_recorder)
            cb_iterate(i, 1 / 6.)

            # prune
            print('--- prune ---')
            self.prune()
            boiler_plate(validate_steps, subgraph_recorder)
            cb_iterate(i, 2 / 6.)

            # consolidate
            print('--- consolidate ---')
            self.consolidate()
            boiler_plate(validate_steps, subgraph_recorder)
            cb_iterate(i, 3 / 6.)

            # connect
            print('--- gaps ---')
            self.connect_leaves()
            boiler_plate(validate_steps, subgraph_recorder)
            cb_iterate(i, 4 / 6.)

            # iteration boiler plate
            if (last_graph is not None and
                        set(last_graph.nodes()) == set(self.graph.nodes()) and
                        set(last_graph.edges()) == set(self.graph.edges())):
                L.warn('No change since last iteration, halting')
                break
            last_graph = self.graph.copy()
            # self.report_card.add_step(self.graph, 'iter {i}'.format(i=i+1))
            cb_iterate(i, 5 / 6.)

            if redraw_callback is not None:
                df = self.report()
                redraw_callback(df)

        if callback:
            callback(1)

        return self.graph

    def report(self):
        """ returns the latest graph object as well as a dataframe with a
        report
        """
        return self.report_card.report(show=False)
        # #report_df = self.report_card.report(show=True)
        # if callback:
        #     callback(0.5)
        # #report_df = self.report_card.save_reports(self.graph)
        # if callback:
        #     callback(1)
        # return self.graph, report_df

    def write_reports(self):
        return self.report_card.save_reports(self.graph)

    def untangle_collsions(self, verbose=True):
        """ attempt to find and untangle collisions in the graph
        """
        graph = self.graph
        experiment = self.experiment
        cr = collider.CollisionResolver(experiment, graph)
        # bounding box method
        print('collisions from bbox')

        # initialize records
        resolved = set()
        overlap_fails = set()
        data_fails = set()
        dont_bother = set()

        # trying_new_suspects = True
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

            report = cr.resolve_overlap_collisions(list(s))

            newly_resolved = set(report['resolved'])
            resolved = resolved | newly_resolved
            collisions_were_resolved = len(newly_resolved) > 0

            # keep track of all fails that had missing data that have
            # not been resolved yet
            data_fails = data_fails | set(report['missing_data'])
            # data_fails = data_fails - resolved

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

            if float(full_count):
                p_res = int(100.0 * n_res / float(full_count))
                p_dat = int(100.0 * len(data_fails) / float(full_count))
                p_no1 = int(100.0 * no1 / float(full_count))
                p_no2 = int(100.0 * no2 / float(full_count))
            else:
                p_res = p_dat = p_no1 = p_no2 = 0.0

            print('\t{n} resolved {p}%'.format(n=n_res, p=p_res))
            print('\t{n} missing data {p}%'.format(n=n_dat, p=p_dat))
            print('\t{n} missing data, no overlap {p}%'.format(n=no1, p=p_no1))
            print('\t{n} full data, no  overlap {p}%'.format(n=no2, p=p_no2))

        self.report_card.add_step(self.graph, step_name='resolve collisions',
                                  phase_name=self.phase_name)
        return len(resolved)


def calculate_duration_data_from_graph(experiment, graph, node_ids=[]):
    if not node_ids:
        node_ids = graph.nodes(data=False)

    frame_times = experiment.frame_times
    step_data, durations = [], []
    for node in node_ids:
        node_data = graph.node[node]
        bf, df = node_data['born_f'], node_data['died_f']
        t0 = frame_times[bf - 1]
        tN = frame_times[df - 1]
        step_data.append({'bid': node, 't0': t0, 'tN': tN, 'lifespan': tN - t0})

    steps = pd.DataFrame(step_data)
    steps.set_index('bid', inplace=True)
    steps = steps / 60.0  # convert to minutes.
    steps.sort('t0', inplace=True)
    steps = steps[['t0', 'tN', 'lifespan']]
    durations = np.array(steps['lifespan'])
    return steps, durations