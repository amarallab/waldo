#!/usr/bin/env python

import math
import profiling

def peter_calculate_midline_from_line(line):
    ''' accepts a single line from a blobs file and returns a length estimate 
    or None.
    '''
    # the midline portion starts with a '%' and ends either with '%%' or line end.
    split_line = line.split('%') 
    if len(split_line) < 2: return None
    if len(split_line) > 4: return None
    cols = split_line[1].split()
    # there are 22 columns (all integers)
    if len(cols) != 22: return None
    # [x1, y1, x2, y2, ... ]
    try: 
        xs = [float(x) for x in cols[::2]]
        ys = [float(y) for y in cols[1::2]]
        midline = 0
        px, py = 0
        for i in range(1, len(xs)):
            cx, cy = xs[i], ys[i]
            midline += math.sqrt((cx - px)**2 + (cy - py)**2)
            px, py = cx, cy
#            midline += math.sqrt((ys[i] - ys[i+1])**2 + (xs[i] - xs[i+1])**2)
        return midline
    except Exception as e:
        print e
        return None

def heltena_calculate_midline_from_line(line):
    split_line = line.split('%')
    if len(split_line) < 2: return None
    if len(split_line) > 4: return None
    cols = split_line[1].split()
    # there are 22 columns (all integers)
    if len(cols) != 22: return None
    # [x1, y1, x2, y2, ... ]
    try:
        midline = 0
        px = float(cols[0])
        py = float(cols[1])
        for i, v in enumerate([float(x) for x in cols[2:]]):
            if i % 2 == 0:
                cx = v
            else:
                cy = v
                midline += math.sqrt((cy - py) ** 2 + (cx - px) ** 2)
                px = cx
                py = cy
        return midline
    except Exception as e:
        print e
        return None

with open("lines_examples.txt", "rt") as f:
    lines = [line[len("HELTENA:  "):] for line in f.readlines()]

ITERATIONS = 1000

profiling.begin("peter")
for i in range(ITERATIONS):
    peter = [peter_calculate_midline_from_line(line) for line in lines]
profiling.end("peter")

profiling.begin("heltena")
for i in range(ITERATIONS):
    heltena = [heltena_calculate_midline_from_line(line)for line in lines]
profiling.end("heltena")

if peter != heltena:
    print "NO SON IGUALES"
    print peter