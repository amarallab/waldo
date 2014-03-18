#!/usr/bin/env python

import profiling
from random import random
import copy

cur_log = []

def log(s):
    global cur_log
    cur_log.append(s)

def equally_space(x, points):
    log("equally_space(%s, %s)" % (x, points))
    return x

def reverse_points_if_backwards(a, b):
    log("reverse_points_if_backwards(%s, %s)" % (a, b))
    return a, False

def test_peter(treated_spines):
    cur_log = []
    treated_spines = map(lambda x: equally_space(x, points=50), treated_spines)

    for i in range(len(treated_spines)):
        if i > 0:
            last_spine = treated_spines[i - 1]
            this_spine = treated_spines[i]
            if this_spine and last_spine:
                assert len(last_spine) == len(this_spine), 'spines unequal len: %i -- %i ' % (len(last_spine),
                                                                                              len(this_spine))
                #print 'spines are len: %i -- %i ' %(len(last_spine), len(this_spine))
                # this makes all spines consistently facing same direction
                final_spine, reversed_flag = reverse_points_if_backwards(last_spine, this_spine)
            else:
                # if both not present, then just use the current spine
                final_spine = this_spine
                #standardized_spines.append(final_spine)
            treated_spines[i] = final_spine
            #x, y = zip(*final_spine)
            #plot(x,y, ls='',marker='o', alpha=0.5)
        log(treated_spines)
        return cur_log

def test_heltena(treated_spines):
    cur_log = []
    treated_spines = map(lambda x: equally_space(x, points=50), treated_spines)
    tmp_spines = [treated_spines[0]]  # first one is not treated
    last_spine = treated_spines[0]
    for this_spine in treated_spines[1:]:
        if this_spine and last_spine:
            assert len(last_spine) == len(this_spine), 'spines unequal len: %i -- %i ' % (len(last_spine),
                                                                                          len(this_spine))
            #print 'spines are len: %i -- %i ' %(len(last_spine), len(this_spine))
            # this makes all spines consistently facing same direction
            final_spine, reversed_flag = reverse_points_if_backwards(last_spine, this_spine)
        else:
            # if both not present, then just use the current spine
            final_spine = this_spine
        last_spine = final_spine
        tmp_spines.append(final_spine)

        #standardized_spines.append(final_spine)
        #x, y = zip(*final_spine)
        #plot(x,y, ls='',marker='o', alpha=0.5)

        #draw()
        #clf()
    treated_spines = tmp_spines
    log(treated_spines)
    return cur_log

data = [[random() * 100] for i in range(200000)]
pd = copy.copy(data)
ph = copy.copy(data)

profiling.begin("peter")
rp = test_peter(pd)
profiling.end("peter")

profiling.begin("heltena")
rh = test_heltena(ph)
profiling.end("heltena")

if rp != rh:
    print "SON DISTINTOS"
