from __future__ import absolute_import

__author__ = 'heltena'

import logging

L = logging.getLogger(__name__)

# Returns true if the group can be collapsed:
#  - the first node must have the predecessors outside the group
#  - the first node must have all the successors inside the group
#  - the last node must have the predessors inside the group
#  - the last node must have the predessors outside the group
#  - the others must have the predessors and the successors inside the group
#  - from the first died and the last born cannot have more time than max_duration
def group_can_be_collapsed(graph, root, group, max_duration):
    first_node = min(group, key=lambda x: graph.node[x]['born_t'])
    if root != first_node:
        return False

    last_node = max(group, key=lambda x: graph.node[x]['died_t'])

    first_died = graph.node[first_node]['died_t']
    last_born = graph.node[last_node]['born_t']
    if last_born - first_died > max_duration:
        return False

    for c in group:
        if c == first_node:
            pred_in_group, succ_in_group = False, True
        elif c == last_node:
            pred_in_group, succ_in_group = True, False
        else:
            pred_in_group, succ_in_group = True, True

        for p in graph.predecessors(c):
            if p in group and not pred_in_group:
                return False
            if p not in group and pred_in_group:
                return False

        for s in graph.successors(c):
            if s in group and not succ_in_group:
                return False
            if s not in group and succ_in_group:
                return False
    return True


# This algorithm looks for a group of nodes that:
#  - all the predecessors are in the group except for the first one
#  - all the succesors are in the group except the last one
#  - the "time" between the "died" value from the first one and the "born"
#    value from the last one is less than max_duration
def collapse_group_of_nodes(graph, max_duration):
    nodes = set(graph.nodes())
    while nodes:
        root = nodes.pop()
        if root != 10:
            continue
        root_died = graph.node[root]['died_t']
        last_born = graph.node[root]['died_t']
        remain = set(graph.successors(root))
        current_group = set([root])

        while remain and last_born - root_died < max_duration:
            current = min(remain, key=lambda x: graph.node[x]['born_t'])
            last_born = max(last_born, graph.node[current]['born_t'])

            remain.remove(current)
            if current in current_group:
                continue

            succ = graph.successors(current)
            remain = remain | set(succ)
            current_group.add(current)

            # test if it is a group that it can be collapsed
            if len(current_group) > 1 and group_can_be_collapsed(graph, root, current_group, max_duration):
                ss = []
                for n in current_group:
                    preds = (str(a) for a in graph.predecessors(n))
                    succs = (str(a) for a in graph.successors(n))
                    ss.append("%d: Pred: %s, Succ: %s" % (n, ", ".join(preds), ", ".join(succs)))
                L.debug("I: Group: (%s)" % ") - (".join(ss))

                current_group.remove(root)
                others = current_group    # All nodes except the root

                graph.condense_nodes(root, *others)
                nodes.add(root)
                nodes |= set(graph.predecessors(root))
                nodes -= others
                break  # collapsed, start again with another node
