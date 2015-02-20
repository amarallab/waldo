from __future__ import absolute_import

__author__ = 'heltena'

import logging

L = logging.getLogger(__name__)

 # This algorithm looks for a group of nodes that:
#  - all the predecessors are in the group except for the first one
#  - all the succesors are in the group except the last one
#  - the "time" between the "died" value from the first one and the "died" value from the last one is less than
#    max_duration
def collapse_group_of_nodes(graph, max_duration):
    nodes = set(graph.nodes())
    while nodes:
        root = nodes.pop()
        L.debug("Current root: {}".format(root))
        root_died = graph.node[root]['died_t']
        last_born = graph.node[root]['died_t']
        remain = graph.successors(root)
        current_group = [root]

        L.debug("1st Successors: {}".format(remain))

        while remain and last_born - root_died < max_duration:
            remain = sorted(remain, key=lambda x: -graph.node[x]['born_t'])
            current = remain.pop()
            if current in current_group:
                continue

            succ = graph.successors(current)
            L.debug("2nd Successors: {}".format(succ))
            if not succ:
                # test if it is a line
                valid = True
                head = root
                line = [root]
                while valid and head != current:
                    succs = graph.successors(head)
                    if len(succs) != 1:
                        valid = False
                    else:
                        head = succs[0]
                        preds = graph.predecessors(head)
                        if len(preds) != 1:
                            valid = False
                        else:
                            line.append(head)
                if valid and head == current:
                    # Yes, it is a line from 'root' to 'current', collapse it
                    L.debug("I: Line between {root} and {current}: {line}".format(root=root, current=current, line=line))
                    graph.condense_nodes(line[0], *line[1:])
                    nodes.add(line[0])
                    nodes |= set(graph.predecessors(line[0]))
                    nodes -= set(line[1:])
                else:
                    L.debug("I: Node {root} has a branch in {current}".format(root=root, current=current))
                break  # branches are not allowed

            remain = list(set(remain) | set(succ))
            current_group.append(current)

            last_born = max(last_born, graph.node[current]['born_t'])

            # test if it is a group that it can be collapsed
            if len(current_group) > 1 and last_born - root_died <= max_duration:
                # test if only two nodes are connected to outside (one using successors, the other using predecessors)

                valid = True
                first_node = None
                last_node = None
                for c in current_group:
                    if len(set(graph.predecessors(c)) - set(current_group)) > 0:
                        if first_node is None:
                            first_node = c
                        else:
                            valid = False
                            break
                    if len(set(graph.successors(c)) - set(current_group)) > 0:
                        if last_node is None:
                            last_node = c
                        else:
                            valid = False
                            break

                if first_node == last_node:
                    valid = False

                if root != first_node:
                    valid = False

                # Ok, collapse!
                if valid:
                    # Monitoring
                    ss = []
                    for n in current_group:
                        preds = (str(a) for a in graph.predecessors(n))
                        succs = (str(a) for a in graph.successors(n))
                        ss.append("%d: Pred: %s, Succ: %s" % (n, ", ".join(preds), ", ".join(succs)))
                    L.debug("I: Group: (%s)" % ") - (".join(ss))

                    graph.condense_nodes(current_group[0], *current_group[1:])
                    nodes.add(current_group[0])
                    nodes |= set(graph.predecessors(current_group[0]))
                    nodes -= set(current_group[1:])
                    break  # collapsed, start again with another node
                else:
                    L.debug("I: Not valid first: {first}, last: {last}, group: {group}".format(first=first_node, last=last_node, group=current_group))
