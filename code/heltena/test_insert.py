#!/usr/bin/env python

from random import random
import copy
import profiling

ITERATIONS = 3000000

data = [random() * 1000 for x in range(ITERATIONS)]
data2 = copy.copy(data)

profiling.begin("Insert")
for i, a in enumerate(data):
	data[i] = a * 2
profiling.end("Insert")

profiling.begin("New array")
data3 = []
for a in data2:
	data3.append(a * 2)
profiling.end("New array")

