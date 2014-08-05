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



import setpath
os.environ.setdefault('WALDO_SETTINGS', 'default_settings')


from conf import settings
import wio.file_manager as fm
from wio.experiment import Experiment
import tape.taper as tp
import collider

DATA_DIR = settings.LOGISTICS['filesystem_data']



ex_id = '20130318_131111'
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

        durations.append(digraph.node[node]['died'] - digraph.node[node]['born'])

    duration_med = np.median(durations)
    duration_std = np.std(durations)

    report = {'total-nodes':graph.number_of_nodes(),
              'isolated-nodes': isolated_count,
              'connected-nodes': connected_count,
              'giant-component-size':giant_size,
              'duration-med': round(duration_med, ndigits=2),
              'duration-std': round(duration_std, ndigits=2),
              '# components': len(components),
              }

    return report






experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
graph = experiment.collision_graph

#graph = pickle.load(open('/home/projects/worm_movement/Data/dev/collider_networks/20130318_131111_graphcache2.pkl'))

reports = []

r = graph_report(graph)
r['step'] = 'raw'
reports.append(r)

collider.remove_nodes_outside_roi(graph, experiment)
r = graph_report(graph)
r['step'] = 'remove outside roi'
reports.append(r)

collider.remove_single_descendents(graph)
r = graph_report(graph)
r['step'] = 'remove single descendents'
reports.append(r)

collider.remove_fission_fusion(graph)
r = graph_report(graph)
r['step'] = 'remove fission-fusion'
reports.append(r)


collider.remove_fission_fusion_rel(graph, split_rel_time=0.5)
r = graph_report(graph)
r['step'] = 'remove fission-fusion relative'
reports.append(r)

collider.remove_offshoots(graph, threshold=20)
r = graph_report(graph)
r['step'] = 'remove offshoots'
reports.append(r)

collider.remove_single_descendents(graph)
r = graph_report(graph)
r['step'] = 'remove single descendents'
reports.append(r)

# threshold = 2
# suspects = collider.suspected_collisions(graph, threshold)
# print(len(suspects), 'suspects found')
# collider.resolve_collisions(graph, experiment, suspects)

columns = ['step', 'total-nodes', 'isolated-nodes',
           'connected-nodes', 'giant-component-size',
           'duration-med', 'duration-std', '# components']


report_card = pd.DataFrame(reports, columns=columns)
report_card.set_index('step')
print report_card

# taper = tp.Taper(experiment=experiment, graph=graph)
# start, end = taper.find_start_and_end_nodes()
# gaps = taper.score_potential_gaps(start, end)
# gt = taper.greedy_tape(gaps, add_edges=False)
# lt = taper.lazy_tape(gaps, add_edges=False)
