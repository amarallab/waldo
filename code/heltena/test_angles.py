#!/usr/bin/env python

import numpy as np
import profiling
from random import random

ITERATIONS = 10000000
data = [random() * 2 * np.pi for i in range(ITERATIONS)]

test = "Mult 180 / np.pi"
profiling.begin(test)
for a in data:
	x = a * 180.0 / np.pi
profiling.end(test)

test = "Mult x = 180 / np.pi"
profiling.begin(test)
f = 180.0 / np.pi
for a in data:
	x = a * f
profiling.end(test)

test = "numpy func"
profiling.begin(test)
for a in data:
	x = np.degrees(a)
profiling.end(test)
 

