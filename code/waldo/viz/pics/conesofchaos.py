# -*- coding: utf-8 -*-
"""
Taper visualizations: what's going on in the "candidate cone" with 10
possible candidates popping up?
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import argparse

# third party
import pandas as pd
import matplotlib.pyplot as plt

# project specific
from waldo import wio

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_id',
        help="Experiment ID to use.")
    parser.add_argument('lost_blob_id', type=int,
        help="Blob ID to look around the end")
    parser.add_argument('-r', '--radius', type=float, default=150,
        help='Pixel radius ("barrel-through-time radius")')
    parser.add_argument('-t', '--times', nargs='+', type=int, default=[-50, 500],
        help='Times relative to the end of the target blob '
             '("barrel-through-time length")')

    args = parser.parse_args()
    lost_id = args.lost_blob_id
    dr = args.radius
    dt = args.times
    if len(dt) == 1:
        dt = 2*dt
    elif len(dt) > 2:
        parser.error('times can only be one or two numbers (interpreted >> {} << )'.format(dt))

    experiment = wio.Experiment(experiment_id=args.experiment_id)

    terminals = experiment.prepdata.load('terminals')
    terminals.set_index('bid', inplace=True)

    try:
        xlost, ylost, flost = terminals.loc[args.lost_blob_id][['xN', 'yN', 'fN']]
    except KeyError:
        parser.exit(message='blob ID >> {} << not present in the terminals '
                            'data summary.\n'.format(args.lost_blob_id))

    lost_blob = experiment[args.lost_blob_id]

    started_near = set(terminals[
        ((terminals.x0 - xlost)**2 + (terminals.y0 - ylost)**2 <= dr**2) &
        (terminals.f0 >= flost + dt[0]) &
        (terminals.f0 <= flost + dt[1])
    ].index)

    ended_near = set(terminals[
        ((terminals.xN - xlost)**2 + (terminals.yN - ylost)**2 <= dr**2) &
        (terminals.fN >= flost + dt[0]) &
        (terminals.fN <= flost + dt[1])
    ].index)

    # combine and make started/ended mutually exclusive
    near = started_near | ended_near
    entirely_near = started_near & ended_near
    started_near = started_near - entirely_near
    ended_near = ended_near - entirely_near
    assert not (started_near & ended_near)
    assert near == entirely_near | started_near | ended_near

    traces = {}
    for bid in near:
        trace = pd.DataFrame(experiment[bid]['centroid'], columns=['x', 'y'])
        trace['f'] = experiment[bid]['frame']
        traces[bid] = trace

    f, ax = plt.subplots()

    for color, bid_set in [('green', started_near), ('red', ended_near), ('blue', entirely_near), ('orange', [lost_id])]:
        for bid in bid_set:
            trace = traces[bid]
            ax.plot(trace.f, trace.x, color=color, marker='o')
            ax.annotate(bid, xy=trace.loc[0][['f', 'x']], xycoords='data')

    xmin = flost + dt[0]
    xmax = flost + dt[1]
    ymin = xlost - dr
    ymax = xlost + dr
    box = [
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymax),
        (xmax, ymin),
        (xmin, ymin),
    ]
    ax.plot(*zip(*box), color='grey', linestyle='--')
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel('Frame')
    ax.set_ylabel('X (pixels)')

    plt.show()
