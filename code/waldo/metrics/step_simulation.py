import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

def naive_simulation(experiment, verbose=False):
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

class IdealPlateSimulation(object):

    def __init__(self, experiment, N0=None):
        self.pool = []
        self.id_counter = 0
        self.start_record = {}
        self.end_record = {}
        self.event_log = None
        self.experiment = experiment

        if N0 is not None:
            self.initialize_pool(N0)

    def initialize_pool(self, N0):
        self.pool = range(N0)
        self.id_counter = 0
        for i in self.pool:
            self.start_record[i] = 0

    def on_event(self, time):
        self.pool.append(self.id_counter)
        self.start_record[self.id_counter] = time
        self.id_counter += 1

    def off_event(self, time):
        random.shuffle(self.pool)
        o = self.pool.pop()
        self.end_record[o] = time

    def end_simulation(self, time):
        for i in self.pool:
            self.end_record[i] = time

    def make_event_log(self):
        losses = self.experiment.prepdata.load('ends')
        gains = self.experiment.prepdata.load('starts')
        off_roi_times = list(losses[losses['reason'] == 'outside-roi']['t'])
        off_edge_times = list(losses[losses['reason'] == 'on_edge']['t'])
        crawl_off = sorted(off_roi_times + off_edge_times)

        on_roi_times = list(gains[gains['reason'] == 'outside-roi']['t'])
        on_edge_times = list(gains[gains['reason'] == 'on_edge']['t'])
        crawl_on = sorted(on_roi_times + on_edge_times)

        print 'off events', len(crawl_off)
        print 'on events', len(crawl_on)
        #print len(set(crawl_off) & set(crawl_on)), 'overlap'

        event_log = {}
        for t in crawl_off:
            time = round(t, ndigits=3)
            events = event_log.get(time, [])
            events.append('off')
            event_log[time] = events

        for t in crawl_on:
            time = round(t, ndigits=3)
            events = event_log.get(time, [])
            events.append('on')
            event_log[time] = events
        self.event_log = event_log

    def estimate_N0(self):
        # get count at begginging of video
        accuracy = self.experiment.prepdata.load('accuracy')
        accuracy.set_index('frame', inplace=True)
        r0 = accuracy.iloc[0]
        N0 = int(r0['false-neg'] + r0['true-pos'])
        #get list of  all events
        return N0

    def run_simulation(self, t0=0, tN=3600):
        times, counts = [], []
        for time, events in self.event_log.iteritems():
            if len(events) > 1:
                random.shuffle(events)
            for event in events:
                if event == 'on':
                    self.on_event(time)
                if event == 'off':
                    self.off_event(time)
            times.append(time)
            counts.append(len(self.pool))
        self.end_simulation(time=tN)
        tc = zip(times, counts)
        times, counts = zip(*sorted(tc))
        return times, counts

    def return_steps(self):
        s = pd.DataFrame(self.start_record, index=['t0'])
        e = pd.DataFrame(self.end_record, index=['tN'])
        print self.id_counter - 1, 'total worm tracks'
        df = pd.concat([s, e]).T
        df['lifespan'] = df['tN'] - df['t0']
        return df

def plot_simulation_comparison(times, counts, experiment):
    #losses = experiment.prepdata.load('end_report')
    #gains = experiment.prepdata.load('start_report')
    accuracy = experiment.prepdata.load('accuracy')
    accuracy.set_index('time', inplace=True)
    wc = accuracy['true-pos'] + accuracy['false-neg']

    fig, ax = plt.subplots()
    ax.plot(wc.index, wc, label='data')
    ax.plot(times, counts, label='sim')
    ax.legend()
    plt.show()

def best_simulation(experiment, verbose=False, N0=None):

    ips = IdealPlateSimulation(experiment)
    ips.make_event_log()
    N0 = ips.estimate_N0()
    print N0, 'initial estimate for worms'
    still_running = True
    while still_running:
        try:
            ips.initialize_pool(N0=N0)
            times, counts = ips.run_simulation(tN=3600)
            still_running = False
        except IndexError as e:
            #print e
            N0 += 1
            print 'increasing estimate to', N0
    #plot_simulation_comparison(times, counts, experiment)
    return ips.return_steps() / 60

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
    return best_simulation(experiment, verbose=verbose)
