__author__ = 'heltena'

from random import random
import itertools
import profiling

values = []
for m in range(-4, 4):
    for x in range(10):
        x = random() * 100
        y = random() * 100
        p = [x, y]
        values.append((m, p))

print len(values)

ITERATIONS = 100

profiling.begin("peter")
for i in range(ITERATIONS):
    s1 = []
    for i, p1 in enumerate(values[:-1]):
        for j, p2 in enumerate(values[i+1:], start=i+1):
            s1.append("%s %s" % (p1, p2))
profiling.end("peter")


profiling.begin("heltena")
s2 = []
for i in range(ITERATIONS):
    for p1, p2 in itertools.combinations(values, 2):
        s2.append("%s %s" % (p1, p2))
profiling.end("heltena")
