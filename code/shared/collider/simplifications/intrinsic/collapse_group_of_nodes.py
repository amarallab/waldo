__author__ = 'heltena'

# This algorithm looks for a group of nodes that:
#  - all the predecessors are in the group except for the first one
#  - all the succesors are in the group except the last one
#  - the "time" between the "died" value from the first one and the "died" value from the last one is less than
#    max_duration
# The algorithm is always keeping the last safe group of nodes that reach the conditions.
# One the candidate list are completed, the algorithm loops the list ordered by inverse length (bigger first). For
# each candidate, remove from the remain list all the lists that contains at least one of the nodes in the candidate.
# The algorithm loops until no candidates found.
def collapse_group_of_nodes(graph, max_duration, debug=False):
    while True:
        candidates = []
        for node in graph.nodes():
            born = graph.node[node]['died_t']
            last_died = graph.node[node]['died_t']
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
                    last_died = max(last_died, graph.node[current]['died_t'])
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
            if debug:
                print("I: Condensing %d group of nodes" % len(result))

            while len(result) > 0:
                r = result.pop(0)
                result = [a for a in result if len(set(a) & set(r)) == 0]
                if debug:
                    ss = []
                    for n in r:
                        preds = (str(a) for a in graph.predecessors(n))
                        succs = (str(a) for a in graph.successors(n))
                        ss.append("%d: Pred: %s, Succ: %s" % (n, ", ".join(preds), ", ".join(succs)))
                    print("I: Group: (%s)" % ") - (".join(ss))
                graph.condense_nodes(r[0], *r[1:])
