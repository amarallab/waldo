#!/usr/bin/env python

# linklist into matrix
#
# Usage:
# <inputfile  linklist2matrix.py [-s] [-N<size>] > outputfile
#
#
# Options:
#
# -s :       symmetrical (i,j,w_ij in linklist implies j,i,w_ij)
#
#
#
# -N<size>:  size of matrix (necessary if matrix has nodes without
#            links. this usually happens with sparse matrices)
#
#
#
# -i :       use given integer ids. Do not treat them as labels.
#            this is only viable if the labels are 0 to N-1
#
#
# Example:
#
# cat linklist.ll | linklist2matrix.py -N512 > matrix.mx
#
#
# Author: Irmak Sirer (2011)
#



import sys
from Shared.Code.Plotting.HeatMap import dictmatrix

symmetrical=False
N = None
useIntLabels = False

translateFile = 'mx_index2label.dict'
translateBackFile = 'mx_label2index.dict'


# FLAGS PASSED ON CMDLINE:
if len(sys.argv) > 1:
    if '-s' in sys.argv:
        #SYMMETRICAL
        symmetrical = True
    for opt in sys.argv[1:]:
        if opt[:2] == '-N':
            N = int(opt[2:])
    if '-i' in sys.argv:
        # USE GIVEN INTEGER IDS
        useIntLabels = True


# INITIALIZE AND SET TOOLS
mx = dictmatrix(0.)
nodeid2no = {}
nodeno2id = {}
nodeno = 0



#READ
for line in sys.stdin:
    elements = line.split()
    if len(elements) == 3:
        ilabel,jlabel,wij = line.split()
    elif len(elements) == 2:
        ilabel,jlabel = elements
        wij = 1
    else:
        raise IndexError("Input File Format Error: Invalid number of columns")
    

    if useIntLabels:
        i = int(ilabel)
        j = int(jlabel)
    else:
        # create a labeldict
        if ilabel in nodeid2no:
            i = nodeid2no[ilabel]
        else:
            i = nodeno
            nodeno += 1
            nodeid2no[ilabel] = i
            nodeno2id[i] = ilabel
        if jlabel in nodeid2no:
            j = nodeid2no[jlabel]
        else:
            j = nodeno
            nodeno += 1
            nodeid2no[jlabel] = j
            nodeno2id[j] = jlabel
        

    wij = float(wij)
    if wij!=0:
        mx.put(i,j,wij)
        if symmetrical:
            mx.put(j,i,wij)

if N is None:
    size = mx.size()
    mx.print_full_matrix(delimiter=" ")
else:
    size = N
    for i in xrange(N):
        for j in xrange(N):
            print '%g' % mx[i][j],
        print

if not useIntLabels:
    f = open(translateFile, 'w')
    for no, label in sorted(nodeno2id.items()):
        print >> f, '%i\t%s' % (no, label)
    f = open(translateBackFile, 'w')
    for label, no in sorted(nodeid2no.items()):
        print >> f, '%s\t%i' % (label, no)

print >> sys.stderr, "<linklist2matrix> Network size: ", size
if not useIntLabels:
    print >> sys.stderr, "index --> nodelabel table written in %s" % translateFile
    print >> sys.stderr, "nodelabel --> index table written in %s" % translateBackFile





