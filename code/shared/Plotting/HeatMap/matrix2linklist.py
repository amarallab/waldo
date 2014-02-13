#!/usr/bin/env python

# Convert adjacency matrix test file into link list text file
# Author: Irmak Sirer, December 2010
#
# Usage: cat adjmatrix.dat | python matrix2linklist.py > linklist.dat
#
#  This will write the weights in float format. Use the option -i to
# get them in integers.
#
# Options (simple parsing, use one option only)
#
# -1     The node labels start from 1 instead of 0
#
# -s     symmetrical matrix (symmetrize if not already symmetrical
#                            and only output i j w, not j i w     )
#
# Example:
# cat adjmatrix.dat | python matrix2linklist.py -1 > linklist.ll
#

import sys

#parse options
symmetrical = False
node_labels_start_from_1 = False
if len(sys.argv) > 1:
    if '-s' in sys.argv:
        symmetrical = True
    if '-1' in sys.argv:
        node_labels_start_from_1 = True
    if '-1s' in sys.argv or '-s1' in sys.argv:
        symmetrical = True
        node_labels_start_from_1 = True
#----


import sys

mx = []
for line in sys.stdin:
    row = map(float, line.split())
    mx.append(row)

# format check
num_rows = len(mx)
for row in mx:
    num_cols = len(row)
    if num_rows != num_cols:
        print >> sys.stderr, "Warning! n_rows != n_cols"

# inform size
size = len(mx)
print >> sys.stderr, "<matrix2linklist> Network size: ", size


# symmetrizing function
def symmetrize(adj, method='average'):
    for i in xrange(size):
        for j in xrange(i+1, size):
            w_ij = adj[i][j]
            w_ji = adj[j][i]
            if method == 'average':
                w_both = (w_ij + w_ji) / 2.
            elif method == 'max':
                w_both = max([w_ij, w_ji])
            else:
                raise ValueError("symmetrize method is either 'average' or 'max'")
            adj[i][j] = adj[j][i] = w_both
    return adj

if symmetrical:
    # symmetrize
    mx = symmetrize(mx, method='average')



if symmetrical:
    # print linklist - symm
    for i in range(size):
        for j in range(i, size):
            linkweight = mx[i][j]
            if linkweight != 0:
                if node_labels_start_from_1:
                    print '%i %i %g' % (i+1,j+1,linkweight)
                else:
                    print '%i %i %g' % (i,j,linkweight)
                # Note on why to use option -1 (to print i+1 and j+1):
                # modules_weight_SA doesn't like 0 as a node name,
                # so shift everything by 1

else:
    # print linklist - asymm
    for i in xrange(size):
        for j in xrange(size):
            linkweight = mx[i][j]
            if linkweight != 0:
                if node_labels_start_from_1:
                    print '%i %i %g' % (i+1,j+1,linkweight)
                else:
                    print '%i %i %g' % (i,j,linkweight)
                # Note on why to use option -1 (to print i+1 and j+1):
                # modules_weight_SA doesn't like 0 as a node name,
                # so shift everything by 1


