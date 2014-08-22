__author__ = 'heltena'

#import pathcustomize
import os
import sys

HERE = os.path.abspath('.')
CODE_DIR = os.path.join(HERE, '..')
print CODE_DIR
sys.path.append(CODE_DIR)
import setpath
import platform
import math

from collapse_group_of_nodes import collapse_group_of_nodes

print('Python {} ({}) [{}] on {}'.format(platform.python_version(), ', '.join(platform.python_build()),
                                         platform.python_compiler(), sys.platform))

from wio.experiment import Experiment
import collider
from conf import settings

SUITE_DEFAULTS = {
    'offshoots': settings.COLLIDER_SUITE_OFFSHOOT,
    'splits_abs': settings.COLLIDER_SUITE_SPLIT_ABS,
    'splits_rel': settings.COLLIDER_SUITE_SPLIT_REL,
    'assimilate': settings.COLLIDER_SUITE_ASSIMILATE_SIZE,
}


def find_potential_collisions(graph, experiment, min_duration, duration_factor):
    candidates = []
    for node in graph.nodes():
        succs = graph.successors(node)
        if len(succs) < 2:
            continue

        preds = graph.predecessors(node)
        if len(preds) < 2:
            continue

        min_succ_duration = None
        for s in succs:
            cur_preds = graph.predecessors(s)
            if len(cur_preds) != 1 or cur_preds[0] != node:
                min_succ_duration = None
                break
            duration = graph.node[s]['died'] - graph.node[s]['born']
            if min_succ_duration is None or min_succ_duration > duration:
                min_succ_duration = duration
        if min_succ_duration is None:
            continue

        min_pred_duration = None
        for p in preds:
            cur_succs = graph.successors(p)
            if len(cur_succs) != 1 or cur_succs[0] != node:
                min_pred_duration = None
                break
            duration = graph.node[p]['died'] - graph.node[p]['born']
            if min_pred_duration is None or min_pred_duration > duration:
                min_pred_duration = duration
        if min_pred_duration is None:
            continue

        duration = graph.node[node]['died'] - graph.node[node]['born']
        if duration >= min_duration and duration < min_succ_duration * duration_factor and duration < min_pred_duration * duration:
            candidates.append(node)
    print "Candidates (%d): %s" % (len(candidates), candidates)
    for n in candidates:
        preds = (str(a) for a in graph.predecessors(n))
        succs = (str(a) for a in graph.successors(n))
        print "%d: Pred: %s, Succ: %s" % (n, ", ".join(preds), ", ".join(succs))
        for a in graph.predecessors(n):
            succ = graph.successors(a)
            if len(succ) != 1 or succ[0] != n:
                print "ERROR!!!!"
        for a in graph.successors(n):
            pred = graph.predecessors(a)
            if len(pred) != 1 or pred[0] != n:
                print "ERROR!!!!"


if __name__ == "__main__":
    ex_id = '20130318_131111'
    experiment = Experiment(experiment_id=ex_id)
    graph = experiment.graph.copy()
    collider.removal_suite(graph)

    params_local = SUITE_DEFAULTS.copy()
    params_local.update()

    collider.remove_single_descendents(graph)

    collider.remove_fission_fusion(graph, max_split_frames=params_local['splits_abs'])
    collider.remove_fission_fusion_rel(graph, split_rel_time=params_local['splits_rel'])

    collider.remove_offshoots(graph, threshold=params_local['offshoots'])
    collider.remove_single_descendents(graph)

    max_duration = 30
    collapse_group_of_nodes(graph, experiment, max_duration)

    min_duration = 1
    duration_factor = 200000000000000
    find_potential_collisions(graph, experiment, min_duration, duration_factor)
