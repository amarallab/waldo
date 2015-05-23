import os

import numpy as np

#from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
#from mpltools import style


#import sys
#sys.path.append('../scripts')
#print sys.path
import pathcustomize

from code.waldo import wio
import waldo.images.evaluate_acuracy as ea
import waldo.images.worm_finder as wf
import waldo.metrics.report_card as report_card

SAVE_DIR = os.path.abspath('../results/fig1-2014-07-31')
plt.style.use('ggplot')

def calculate_stats_for_moves(min_moves, prep_data):
    move = prep_data.load('moved')

    base_accuracy = prep_data.load('accuracy', index_col=False)
    print(base_accuracy.head())
    print(base_accuracy.columns)
    matches = prep_data.load('matches')
    counts, tps, fps, fns = [], [], [], []
    for mm in min_moves:
        print(mm)
        moved = move[move['bl_moved'] >= mm]
        bids = list(moved['bid'])
        filtered_accuracy = ea.recalculate_accuracy(matches, base_accuracy, bids=bids)
        #filtered_accuracy = filtered_accuracy[np.isfinite(filtered_accuracy['true-pos'])]
        counts.append(len(bids))
        tp = filtered_accuracy['true-pos'].mean()
        #print 'tp', tp
        fp = filtered_accuracy['false-pos'].mean()
        #print 'fp', fp
        fn = filtered_accuracy['false-neg'].mean()

        tps.append(tp)
        fps.append(fp)
        fns.append(fn)

    tps = np.array(tps)
    fps = np.array(fps)
    fns = np.array(fns)
    totals = fns + tps
    tps_p = tps / totals * 100
    fps_p = fps / totals * 100
    fns_p = fns / totals * 100

    print('true counts=', np.mean(totals))

    data = pd.DataFrame([tps_p, fns_p, fps_p], index=['TP', 'FN', 'FP']).T
    counts = pd.DataFrame(counts, columns=['counts'])
    return data, counts


# def show_mid_long(dfs, labels=['raw', 'final'], ts=[5,30], axes=None):
#     df, df2 = dfs
#     t1, t2 = ts

#     def new_plot(ax, df, label):
#         step_df = df
#         n_steps = len(step_df)
#         xmax = 60
#         ymax = n_steps + 1
#         steps = []

#         xs = list(step_df['t0'])
#         widths = list(step_df['lifespan'])
#         height = 1

#         color = ax._get_lines.color_cycle.next()
#         for y, (x, width) in enumerate(zip(xs, widths)):
#             steps.append(patches.Rectangle((x,y), height=height, width=width,
#                                            fill=True, fc=color, ec=color, alpha=0.5))
#         for step in steps:
#             ax.add_patch(step)

#         ax.plot([0], color=color, label=label, alpha=0.5)
#         ax.set_xlim([0, xmax])
#         #ax.set_ylim([0, ymax])
#         ax.set_xlabel('t (min)')

#     if axes == None:
#         fig, ax = subplots(2,1)
#     ax1, ax2 = axes
#     mid1 = df[(df['lifespan'] < t2) & (df['lifespan'] >= t1)]
#     mid2 = df2[(df2['lifespan'] < t2) & (df2['lifespan'] >= t1)]
#     new_plot(ax1, mid2, 'final')
#     new_plot(ax1, mid1, 'raw')
#     plt.legend(loc='upper left')

#     fig, ax = plt.subplots()
#     long1 = df[df['lifespan'] >= t2]
#     long2 = df2[df2['lifespan'] >= t2]
#     new_plot(ax2, long2, 'final')
#     new_plot(ax2, long1, 'raw')
#     plt.legend(loc='upper left')


def make_axes_square(ax):
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))


def step_facets(df, df2, t1=5, t2=20):

    def new_plot(ax, df, label, nth_color=0, xmax=60):
        step_df = df
        n_steps = len(step_df)
        steps = []


        xs = list(step_df['t0'])
        widths = list(step_df['lifespan'])
        height = 1

        color = ax._get_lines.color_cycle.next()
        for i in range(nth_color):
            color = ax._get_lines.color_cycle.next()

        for y, (x, width) in enumerate(zip(xs, widths)):
            steps.append(patches.Rectangle((x,y), height=height, width=width,
                                           fill=True, fc=color, ec=color, alpha=0.5))
        for step in steps:
            ax.add_patch(step)

        ax.plot([0], color=color, label=label, alpha=0.5)
        ax.set_xlim([0, xmax])
        #ax.set_ylim([0, ymax])

    xmax = max([max(df['tN']), max(df['tN'])])

    #fig = plt.Figure(figsize=(1, 1))

    gs = gridspec.GridSpec(3, 2)
    gs.update(wspace=0.01, hspace=0.05) #, bottom=0.1, top=0.7)

    ax_l0 = plt.subplot(gs[0,0])
    ax_l1 = plt.subplot(gs[0,1], sharey=ax_l0)
    ax_m0 = plt.subplot(gs[1,0])
    ax_m1 = plt.subplot(gs[1,1], sharey=ax_m0)
    ax_s0 = plt.subplot(gs[2,0])
    ax_s1 = plt.subplot(gs[2,1], sharey=ax_s0)

    left = [ax_l0, ax_m0, ax_s0]
    right = [ax_l1, ax_m1, ax_s1]
    top = [ax_l0, ax_l1]
    mid = [ax_m0, ax_m1]
    bottom = [ax_s0, ax_s1]
    ylabels = ['> {t2} min'.format(t2=t2), '{t1} to {t2} min'.format(t1=t1, t2=t2), '< {t1} min'.format(t1=t1)]


    short1 = df[df['lifespan'] < t1]
    short2 = df2[df2['lifespan'] < t1]
    mid1 = df[(df['lifespan'] < t2) & (df['lifespan'] >= t1)]
    mid2 = df2[(df2['lifespan'] < t2) & (df2['lifespan'] >= t1)]
    long1 = df[df['lifespan'] >= t2]
    long2 = df2[df2['lifespan'] >= t2]

    short_max = max([len(short1), len(short2)])
    mid_max = max([len(mid1), len(mid2)])
    long_max = max([len(long1), len(long2)])
    print('short', short_max)
    print('long', long_max)
    print('mid', mid_max)
    print(long1.head(20))

    new_plot(ax_s0, short1, 'raw', xmax=xmax)
    new_plot(ax_m0, mid1, 'raw', xmax=xmax)
    new_plot(ax_l0, long1, 'raw', xmax=xmax)

    new_plot(ax_s1, short2, 'final', nth_color=1, xmax=xmax)
    new_plot(ax_m1, mid2, 'final', nth_color=1, xmax=xmax)
    new_plot(ax_l1, long2, 'final', nth_color=1, xmax=xmax)

    for ax in right:
        ax.axes.yaxis.tick_right()

    for ax, t in zip(top, ['raw', 'final']):
        #ax.axes.xaxis.tick_top()
        ax.get_xaxis().set_ticklabels([])
        ax.set_ylim([0, long_max])
        #ax.legend(loc='upper left', fontsize=20)
        #make_axes_square(ax)
        #ax.set_aspect(1)

    for ax in mid:
        ax.get_xaxis().set_ticklabels([])
        ax.set_ylim([0, mid_max])
        #ax.set_aspect(1)
        #make_axes_square(ax)

    for ax in bottom:
        ax.set_xlabel('t (min)')
        ax.set_ylim([0, short_max])
        #make_axes_square(ax)
        #ax.set_aspect(1)


    for ax, t in zip(left, ylabels):
        ax.set_ylabel(t)

    ax_s0.get_xaxis().set_ticklabels([0, 10, 20, 30, 40, 50])

def steps_from_node_report(experiment, min_bl=1):
    node_report = experiment.prepdata.load('node-summary')
    print(node_report.head())
    steps = node_report[['bl', 't0', 'tN', 'bid']]
    steps.set_index('bid', inplace=True)

    steps['t0'] = steps['t0'] / 60.0
    steps['tN'] = steps['tN'] / 60.0
    steps['lifespan'] = steps['tN'] - steps['t0']
    steps['mid'] = (steps['tN']  + steps['t0']) / 2.0
    return steps[steps['bl'] >= 1]

def make_fig1(ex_id, step_min_move = 1, save_fig=False):

    experiment = wio.Experiment(experiment_id=ex_id)
    #prep_data = experiment.prepdata
    graph = experiment.graph.copy()

    moving_nodes =  [int(i) for i in graph.compound_bl_filter(experiment, threshold=step_min_move)]
    steps, durations = report_card.calculate_duration_data_from_graph(experiment, graph, moving_nodes)

    final_steps = steps_from_node_report(experiment)
    accuracy = experiment.prepdata.load('accuracy')
    worm_count = np.mean(accuracy['true-pos'] + accuracy['false-neg'])
    print('worm count:', worm_count)

    fig, ax = plt.subplots()
    ### AX 0
    # print 'starting ax 0'
    color_cycle = ax._get_lines.color_cycle
    for i in range(5):
        c = color_cycle.next()
    wf.draw_minimal_colors_on_image_T(ex_id, time=30*30, ax=ax, color=c)

    if save_fig:
        plt.savefig('fig1_{eid}_plate.png'.format(eid=ex_id), format='png')
    fig, ax = plt.subplots()
    step_facets(steps, final_steps)
    if save_fig:
        plt.savefig('fig1_{eid}_tracks.png'.format(eid=ex_id), format='png')

    if not save_fig:
        plt.show()

# def make_fig1_old(ex_id, save_name=None):

#     prep_data = wio.file_manager.PrepData(ex_id)

#     # for panel 1
#     min_moves = [0, 0.25, 0.5, 1, 2]
#     data, counts = calculate_stats_for_moves(min_moves, prep_data)
#     steps, durations = calculate_duration_data(prep_data, min_move=2)
#     #return steps
#     #columns=min_moves).T
#     #print data

#     fig = plt.figure()
#     gs = gridspec.GridSpec(4, 7)
#     gs.update(left=0.05, wspace=1.0, hspace=1.0, right=0.95)
#     ax0 = plt.subplot(gs[:,:3])
#     ax1 = plt.subplot(gs[0:2,3:5])
#     ax2 = plt.subplot(gs[0:2,5:])
#     ax3 = plt.subplot(gs[2:,3:5])
#     ax4 = plt.subplot(gs[2:,5:])

#     ### AX 0
#     color_cycle = ax0._get_lines.color_cycle

#     c = {'tp_color':color_cycle.next(),
#          'missed_color': color_cycle.next(),
#          'fp_color': color_cycle.next(),
#          'roi_color': color_cycle.next(),
#          'roi_line_color': color_cycle.next()}
#     wf.draw_colors_on_image(ex_id, time=30*60, ax=ax0, colors=c)

#     ### AX 1
#     min_moves = [0, 0.25, 0.5, 1, 2]
#     for mm in min_moves:
#         steps, durations = calculate_duration_data(prep_data, min_move=mm)
#         ecdf = ECDF(durations)
#         cdf = ecdf(range(65))
#         ax1.plot(cdf, label='move > {m} BL'.format(m=mm))
#     ax1.plot([0, 60, 60,65], [0,0, 1,1], label='ideal')
#     ax1.legend(loc='lower center')
#     ax1.set_ylim([0, 1.001])
#     ax1.set_ylabel('CDF')
#     ax1.set_xlabel('track duration (min)')

#     ### AX 2
#     counts.plot(kind='bar', ax=ax2)
#     ax2.set_ylabel('# tracks')
#     ax2.set_xticklabels([str(round(i,2)) for i in (min_moves)])
#     ax2.set_xlabel('min. body-lengths moved')

#     ### AX 3
#     data.plot(kind='bar', ax=ax3, grid=True)
#     ax3.set_ylabel('percent of true count (TP + FN)')
#     ax3.set_ylim([0,100])
#     ax3.set_xticklabels([str(round(i,2)) for i in (min_moves)])
#     ax3.set_xlabel('min body-lengths moved')
#     #ax3.legend(loc='lower right')

#     ### AX 4

#     step_plot(ax4, steps)
#     ax4.set_ylabel('track number')
#     ax4.set_xlabel('time (min)')
#     ax4.legend(loc='upper left')
#     #plt.tight_layout()

#     if save_name is None:
#         plt.show()
#     else:
#         plt.savefig(fig_name, format='pdf')
# for ex_id in ex_ids:
#     fig_name = '{d}/{eid}-fig1.pdf'.format(d=SAVE_DIR, eid=ex_id)
#     print fig_name
#     fm.ensure_dir_exists(SAVE_DIR)
#     try:
#         make_fig1(ex_id, save_name=fig_name)
#     except Exception as e:
#         print e

if __name__ == '__main__':

    ex_id = '20130414_140704'
    ex_id = '20130614_120518'
    ex_id = '20130318_131111'
    ex_id = '20130702_135704' # testset
    #ex_id = '20130702_135652'

    # new small plates
    ex_id = '20141017_113435'
    ex_id = '20141017_113439'
    ex_id = '20141017_123722'
    #ex_id = '20141017_123725'
    # ex_id = '20141017_151002'
    ex_id = '20130410_165326'
    s = make_fig1(ex_id, save_fig=False)
    #ex_id = '20141017_134724'
    # ex_ids = [
    #     '20141017_134720',
    #     '20141017_134724',
    #     '20130318_131111',
    #     '20130614_120518',
    #     '20130702_135704',
    # ]
    # for ex_id in ex_ids:
    #     try:
    #         s = make_fig1(ex_id, save_fig=True)
    #     except Exception as e:
    #         print 'failed', ex_id, e
