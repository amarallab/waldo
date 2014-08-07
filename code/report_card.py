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



ex_id = '20130318_131111'
ex_id = '20130614_120518'
#ex_id = '20130702_135704' # many pics
# ex_id = '20130614_120518'

#path = os.path.join(DATA_DIR, ex_id)
#print path

def graph_report(digraph):
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


experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
graph = experiment.collision_graph

#graph = pickle.load(open('/home/projects/worm_movement/Data/dev/collider_networks/20130318_131111_graphcache2.pkl'))

show_full = False

############### iniitalization
reports = []
durations = []

r,d = graph_report(graph)
r['step'] = 'raw'
reports.append(r)
print 'raw', r['total-nodes']

durations.append(('raw', d))


collider.remove_nodes_outside_roi(graph, experiment) # <-----
r,d = graph_report(graph)
r['step'] = 'remove outside roi'
reports.append(r)
print 'roi', r['total-nodes']

collider.remove_blank_nodes(graph, experiment) # <-----
r,d = graph_report(graph)
r['step'] = 'remove blank nodes'
reports.append(r)
print 'blank', r['total-nodes']

durations.append(('pruned', d))

############### simplifications
collider.assimilate(graph, max_threshold=10)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'assimilate'
    reports.append(r)

collider.remove_single_descendents(graph) # < --------
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'remove single descendents'
    reports.append(r)

collider.remove_fission_fusion(graph)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'remove fission-fusion'
    reports.append(r)

collider.remove_fission_fusion_rel(graph, split_rel_time=0.5)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'remove fission-fusion relative'
    reports.append(r)

collider.remove_offshoots(graph, threshold=20)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'remove offshoots'
    reports.append(r)

collider.remove_single_descendents(graph)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'remove single descendents'
    reports.append(r)

if not show_full:
    r,d = graph_report(graph)
    r['step'] = 'network_simplifications'
    reports.append(r)

print 'simplified', r['total-nodes']
durations.append(('simplified', d))

############### collisions

# threshold = 2
# suspects = collider.suspected_collisions(graph, threshold)
# print(len(suspects), 'suspects found')
# collider.resolve_collisions(graph, experiment, suspects)
# r,d = graph_report(graph)
# r['step'] = 'basic collisions removed'
# print 'collision', r['total-nodes']

taper = tp.Taper(experiment=experiment, graph=graph)
start, end = taper.find_start_and_end_nodes()
gaps = taper.score_potential_gaps(start, end)
gt = taper.greedy_tape(gaps, threshold=0.001, add_edges=True)
graph = taper._graph
r,d = graph_report(graph)
r['step'] = 'greedy gaps'
reports.append(r)
print 'greedy gap bridging', r['total-nodes']

durations.append(('gaps', d))

############### simplifications
collider.assimilate(graph, max_threshold=10)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'assimilate'
    reports.append(r)

collider.remove_single_descendents(graph)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'remove single descendents'
    reports.append(r)

collider.remove_fission_fusion(graph)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'remove fission-fusion'
    reports.append(r)


collider.remove_fission_fusion_rel(graph, split_rel_time=0.5)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'remove fission-fusion relative'
    reports.append(r)

collider.remove_offshoots(graph, threshold=20)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'remove offshoots'
    reports.append(r)

collider.remove_single_descendents(graph)
if show_full:
    r,d = graph_report(graph)
    r['step'] = 'remove single descendents'
    reports.append(r)

if not show_full:
    r,d = graph_report(graph)
    r['step'] = 'network_simplifications'
    reports.append(r)

print 'simplified', r['total-nodes']
durations.append(('simplifed 2', d))

############### collisions

# threshold = 2
# suspects = collider.suspected_collisions(graph, threshold)
# print(len(suspects), 'suspects found')
# collider.resolve_collisions(graph, experiment, suspects)
# r,d = graph_report(graph)
# r['step'] = 'basic collisions removed'
# print 'collision', r['total-nodes']

############### gaps

taper = tp.Taper(experiment=experiment, graph=graph)
start, end = taper.find_start_and_end_nodes()
gaps = taper.score_potential_gaps(start, end)
gt = taper.greedy_tape(gaps, threshold=0.001, add_edges=True)
graph = taper._graph
r,d = graph_report(graph)
r['step'] = 'greedy gaps'
reports.append(r)
print 'greedy gap bridging', r['total-nodes']
durations.append(('gaps 2', d))

columns = ['step', 'total-nodes', 'isolated-nodes',
           'connected-nodes', 'giant-component-size',
           'duration-med', 'duration-std', '# components']

report_card = pd.DataFrame(reports, columns=columns)
report_card.set_index('step')
print report_card[['step', 'total-nodes', 'isolated-nodes', 'duration-med']]

fig, ax = plt.subplots()
labels = ['raw', 'pruned', 'simplified', 'gaps', 'simplified', 'gaps']

x = np.arange(0, 30, 0.05)
for i, (l, d) in enumerate(durations):
    dur = np.array(d) / 60.0
    ecdf = ECDF(dur)
    cdf = ecdf(x)
    ax.plot(x, cdf, label=l)
ax.legend(loc='lower right')
plt.show()
