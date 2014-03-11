__author__ = 'heltena'

from random import random
import math
import profiling

EPSILON = 1e-6

def peter_points_along_straight_line(m, p1):
    xs = []
    ys = []
    for k in [-0.2, 0.2]:
        xs.append(p1[0] + k)
        ys.append(p1[1] + m * (xs[-1]-p1[0]))
    return xs, ys

def heltena_points_along_straight_line(m, p1):
    x, y = p1
    xs = [x - 0.2, x + 0.2]
    ys = [y - m * 0.2, y + m * 0.2]
    return xs, ys

values = []
for m in range(-4, 4):
    for x in range(100000):
        x = random() * 100
        y = random() * 100
        p = [x, y]
        values.append( (m, p) )

profiling.begin("Peter")
for m, p in values:
    px, py = peter_points_along_straight_line(m, p)
profiling.end("Peter")

profiling.begin("Heltena")
for m, p in values:
    hx, hy = heltena_points_along_straight_line(m, p)
profiling.end("Heltena")

#    if sum([math.fabs(a - b) + math.fabs(c - d) for a, b, c, d in zip(px, hx, py, hy)]) > EPSILON:
#        print "ERROR!: %s != %s" % ( (px, py), (hx, hy))