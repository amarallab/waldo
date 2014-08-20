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


# This algorithm looks for a group of nodes that:
#  - all the predecessors are in the group except for the first one
#  - all the succesors are in the group except the last one
#  - the "time" between the "died" value from the first one and the "died" value from the last one is less than
#    max_duration
# The algorithm is always keeping the last safe group of nodes that reach the conditions.
# One the candidate list are completed, the algorithm loops the list ordered by inverse length (bigger first). For
# each candidate, remove from the remain list all the lists that contains at lest one of the nodes in the candidate.
# The algorithm loops until no candidates found.
def collapse_group_of_nodes(graph, experiment, max_duration):
    while True:
        candidates = []
        for node in graph.nodes():
            born = graph.node[node]['died']
            last_died = graph.node[node]['died']
            remain = graph.successors(node)
            current_group = [node]
            last_safe = None
            while len(remain) > 0 and last_died - born < max_duration:
                current = remain.pop()
                pred = graph.predecessors(current)
                if len(set(pred) - set(current_group)) > 0:
                    break
                succ = graph.successors(current)
                if len(succ) > 0 and current not in current_group:
                    for s in succ:
                        if s not in remain and s not in current_group:
                            remain.append(s)
                    last_died = max(last_died, graph.node[current]['died'])
                    current_group.append(current)
                if len(remain) == 1 and len(current_group) > 1 and last_died - born < max_duration:
                    last_safe = list(current_group)
                    last_safe.append(remain[0])
            if last_safe is not None:
                candidates.append(last_safe)

        result = []
        candidates = sorted(candidates, key=lambda x: len(x))
        while len(candidates) > 0:
            current = candidates.pop()
            result.append(current)
            candidates = [c for c in candidates if len(set(c) & set(current)) == 0]
        if len(result) == 0:
            break
        else:
            print "Condensing %d group of nodes" % len(result)
            for r in result:
                ss = []
                for n in r:
                    preds = (str(a) for a in graph.predecessors(n))
                    succs = (str(a) for a in graph.successors(n))
                    ss.append("%d: Pred: %s, Succ: %s" % (n, ", ".join(preds), ", ".join(succs)))
                print "Group: (%s)" % ") - (".join(ss)
                graph.condense_nodes(r[0], *r[1:])

if __name__ == "__main__":
    ex_id = '20130318_131111'
    experiment = Experiment(experiment_id=ex_id)
    graph = experiment.collision_graph
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
