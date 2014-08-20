def suspected_collisions(digraph, threshold=2):
    """ returns a list of node ids that are suspected collisions.

    suspected collisions are found if the average lifespan of parents
    and the average lifespand of childeren are both [threshold]
    fold longer than the potential suspect.

    parent_1   parent_2
            \   /
             \ /
       potential suspect   | t
             / \           |
            /   \          V
      child_1    child_2

    params
    -----
    digraph: (networkx directed graph object)

    threshold: (int or float)
       the minimum fold difference in lifespan between parents/suspect
       and children/suspect.

    returns
    -----
    suspects: (list)
       all node ids that match the ciriterion of 'suspects'

    """
    print('you are, in fact, running suspected_collisions')
    suspects = []
    for node in digraph:
        #print(node)
        parents = digraph.predecessors(node)
        children = digraph.successors(node)
        if len(parents) != 2 or len(children) != 2:
            continue

        node_life = digraph.lifespan(node)
        parents_life = [digraph.lifespan(p) for p in parents]
        children_life = [digraph.lifespan(c) for c in children]

        #if (sum(parents_life) + sum(children_life)) / (4 * node_life) > threshold:
        if (sum(parents_life) / (2 * node_life) > threshold and
            sum(children_life) / (2 * node_life) > threshold):
            suspects.append(node)

    return suspects
