import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpltools import style

def add_simulation_lines(ax, dfs, n=1, ymax=None, labels=['simulated begining tracks', 'simulated end tracks'], **kwargs):
    """
    """
    color = 'black'
    if 'color' in kwargs:
        color = kwargs.pop('color')

    def lines_from_simulation(dfs, short_lim=5, front_back_margin=10, max_tracks=100):
        front_lim = front_back_margin
        back_lim = round(max(dfs['tN'])) - front_back_margin

        # remove short tracks
        a = dfs[dfs['lifespan'] > short_lim]

        # pull out times from front
        front = a[a['t0'] <= front_lim]
        a = a[a['t0'] > front_lim]
        front.sort('tN', inplace=True, ascending=False)
        f_times = list(front['tN'])

        # calculate times for back tracks
        back = a[a['tN'] >= back_lim]
        a = a[a['tN'] < back_lim]
        back.sort('t0', inplace=True, ascending=True)
        b_times = list(back['t0'])

        return f_times, b_times

    ######## code starts here ########

    f_times, b_times = lines_from_simulation(dfs, **kwargs)
    if ymax is None:
        ymax = len(dfs[dfs['lifespan'] > short_lim])

    front_xy = [(t, i) for (i, t) in enumerate(f_times)]
    back_xy = [(t, ymax - i) for (i, t) in enumerate(b_times)]

    front_buffer = (0, front_xy[-1][1] + 1)
    back_buffer = (60, back_xy[-1][1] - 1)
    front_xy = front_xy + [front_buffer]
    back_xy = back_xy + [back_buffer]

    #print front_xy
    f_x, f_y = zip(*front_xy)
    b_x, b_y = zip(*back_xy)


    ax.plot(f_x, f_y, drawstyle='steps', label=labels[0], color=color)
    ax.plot(b_x, b_y, drawstyle='steps-pre', label=labels[1], color=color, linestyle='--')
    #ax.set_ylim([0, ymax])
    #print 'front xy\n', front_xy

# def calculate_duration_data(prep_data, min_move=2):
#     move = prep_data.load('moved')
#     terminals = prep_data.load('terminals')[['bid', 't0', 'tN']]
#     terminals.set_index('bid', inplace=True)
#     terminals.dropna(inplace=True)
#     #print terminals.head()

#     moved_bids = set(move[move['bl_moved'] >= min_move]['bid'])
#     term_bids = set(terminals.index)
#     term_overlap = term_bids & moved_bids
#     missing_bids = moved_bids - term_overlap
#     print(len(missing_bids), 'moved bids have no terminals')


#     steps = terminals.loc[list(moved_bids)]
#     steps = steps / 60.0
#     steps.sort('t0', inplace=True)
#     steps['lifespan'] = steps['tN'] - steps['t0']
#     durations = np.array(steps['lifespan'])

#     print(len(terminals), 'terminals')
#     print(len(moved_bids), 'moved')
#     print(len(steps), 'steps')
#     return steps, durations

# def step_plot(ax, step_df, color=1, label='moved > 2 BL', ylim=None):
#     steps = []
#     n_steps = len(step_df)

#     xs = list(step_df['t0'])
#     widths = list(step_df['lifespan'])
#     height = 1

#     color_cycle = ax._get_lines.color_cycle
#     for i in range(color):
#         color = color_cycle.next()
#     #color2 = color_cycle.next()
#     for y, (x, width) in enumerate(zip(xs, widths)):
#         steps.append(patches.Rectangle((x,y), height=height, width=width,
#                                        fill=True, fc=color, ec=color))
#     for step in steps:
#         ax.add_patch(step)

#     xmax = 60
#     ax.plot([0], color=color, label=label)
#     ax.set_xlim([0, xmax])
#     if ylim is None:
#         ylim = n_steps + 1
#     ax.set_ylim([0, ylim])



def eye_plot(ax, df, color=None, label='', ymax=None, short_lim=5, front_back_margin=10):
    def plot_patches(ax, df, ymax=ymax, dividers=[], color=color, label=label):
        step_df = df
        n_steps = len(step_df)
        xmax = int(max(df['tN']))
        if ymax is None:
            ymax = n_steps + 1
        steps = []

        xs = list(step_df['t0'])
        widths = list(step_df['lifespan'])
        height = 1
        if color==None:
            c = ax._get_lines.color_cycle.next()
        else:
            for i in range(color):
                c = ax._get_lines.color_cycle.next()

        for y, (x, width) in enumerate(zip(xs, widths)):
            steps.append(patches.Rectangle((x,y), height=height, width=width,
                                           fill=True, fc=c, ec=c))
        for step in steps:
            ax.add_patch(step)

        for d in dividers:
            ax.plot([0, xmax], [d,d], color='black')

        ax.plot([0], color=c, label=label)
        ax.set_xlim([0, xmax])
        ax.set_ylim([0, ymax])
        ax.set_xlabel('t (min)')


    def make_dividers(step_dfs):
        dividers = [len(i) for i in step_dfs]
        dividers = [0] + dividers
        return np.cumsum(dividers)

    ##### code starts here #####


    a = df.sort('lifespan', ascending=False)
    a['mid'] = (a['tN'] + a['t0'])/2
    a = a[a['lifespan'] > short_lim]


    front_lim = front_back_margin
    back_lim = round(max(df['tN'])) - front_back_margin

    all_ids = set(a.index)

    front = a[(a['t0'] <= front_lim) & (a['lifespan'] > 0)]
    front_ids = set(front.index)
    remaining_ids = all_ids - front_ids
    a = a.loc[list(remaining_ids)]
    #a = a[a['t0'] > front_lim]
    front.sort('tN', inplace=True, ascending=False)


    back = a[(a['tN'] >= back_lim) & (a['lifespan'] > 0)]
    back_ids = set(back.index)
    # back_ids = set
    remaining_ids = remaining_ids - back_ids
    a = a.loc[list(remaining_ids)]

    #a = a[a['tN'] < back_lim]
    back.sort('t0', inplace=True, ascending=False)

    mid = a
    mid.sort('mid', inplace=True, ascending=False)


    steps = [front, mid, back]
    #steps = [longest, front, mid, back, shortest]
    a = pd.concat(steps)
    div = make_dividers(steps)
    plot_patches(ax,a, ymax=ymax, dividers=div, color=color, label=label)
    #return div

    legend_props = {'loc':'upper right'}
    if len(div) > 2:
        legend_props = {'bbox_to_anchor':(1, div[-2] + 1), 'bbox_transform':ax.transData, 'loc':'lower left'}
    return legend_props

# def ideal_step_plot(ax, n, xmax=60):
#     n_steps = int(n)
#     color_cycle = ax._get_lines.color_cycle
#     color1 = color_cycle.next()
#     color2 = color_cycle.next()

#     xmax = 60
#     for y in range(n_steps):
#         ideal = patches.Rectangle((0,y), height=1, width=xmax,
#                               fill=True, ec='black', fc=color2,
#                               alpha=0.5)
#         ax.add_patch(ideal)
#     ax.plot([0], color=color2, alpha=0.5, label='ideal')
#     ax.set_xlim([0, xmax])
#     ax.set_ylim([0, n_steps+1])
