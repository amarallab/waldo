import pandas as pd
import numpy as np
import random

def run_ideal_step_simulation(experiment, verbose=False):
    """
    returns a step dataframe by runing a simulation of
    ideal duration counts taking into account the
    - number of worms
    - rate of crawling off a plate
    - rate of crawling onto a plate
    taken from an experiments data

    the order of 'off' and 'on' events are randomized at each
    timestep.
    """

    losses = experiment.prepdata.load('end_report')
    gains = experiment.prepdata.load('start_report')
    accuracy = experiment.prepdata.load('accuracy')

    worm_count = np.mean(accuracy['true-pos'] + accuracy['false-neg'])
    crawl_off_total = (losses['on_edge'] + losses['outside-roi']).sum()
    crawl_on_total = (gains['on_edge'] + gains['outside-roi']).sum()

    if verbose:
        print 'loading data for', experiment.id
        print worm_count, 'worms'
        print crawl_off_total, 'crawl off'
        print crawl_on_total, 'crawl on'

    N = int(worm_count)
    t_steps = 60
    on_rate = crawl_on_total / t_steps
    off_rate = crawl_off_total / t_steps

    start_record = {}
    end_record = {}
    id_counter = N

    # initialize pool and start record
    pool = range(N)
    for i in pool:
        start_record[i] = 0


    off_events = ['off' for i in range(off_rate)]
    on_events = ['on' for i in range(on_rate)]
    events = off_events + on_events

    # start simulation
    for t in range(t_steps):
        random.shuffle(events)
        for e in events:
            random.shuffle(pool)
            if e == 'off':
                o = pool.pop()
                #print 'removing', o, 'at', t
                end_record[o] = t
            if e == 'on':
                #print 'adding', id_counter, 'at', t
                pool.append(id_counter)
                start_record[id_counter] = t
                id_counter += 1

    # get
    for i in pool:
        end_record[i] = t_steps
    s = pd.DataFrame(start_record, index=['t0'])
    e = pd.DataFrame(end_record, index=['tN'])
    print id_counter - 1, 'total worm tracks'
    #print s
    df = pd.concat([s, e]).T
    df['lifespan'] = df['tN'] - df['t0']
    return df
