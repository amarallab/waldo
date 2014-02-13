#!/usr/bin/env python


import sys
from pylab import *

if len(sys.argv) >= 2:
    outfilename=sys.argv[1]
    if len(sys.argv) >= 3:
        colorscheme = sys.argv[2]

# INPUT MATRIX
mx = []
for rowline in sys.stdin:
    row = map(float, rowline.split())
    mx.append(row)


# format check
num_rows = len(mx)
for row in mx:
    num_cols = len(row)
    if num_rows != num_cols:
        print >> sys.stderr, "Warning! n_rows != n_cols"

# inform size
size = len(mx)
print >> sys.stderr, "<plot_matrix> Network size: ", size


# plot
imshow(mx, cmap=cm.YlOrRd, origin='lower')

# (alternative, slower way with pcolor instead of imshow)
# N = size
# x = arange(N)
# y = arange(N)
# X,Y = meshgrid(x,y)
# pcolor(X,Y,mx, cmap=cm.YlOrRd)
# axis([0,N,0,N])

colorbar()
savefig('colormap.pdf')
#show()





