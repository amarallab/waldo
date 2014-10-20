import pathcustomize
import os

import numpy as np
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.path as path
from mpltools import style
from mpltools import layoutsp
import waldo.metrics.step_simulation as ssim

style.use('ggplot')

import sys
sys.path.append('../scripts')
print sys.path
import pathcustomize

import waldo.wio as wio
#import wio.file_manager as fm
#import wio2
import waldo.images.evaluate_acuracy as ea
import waldo.images.worm_finder as wf
#import json

#import multiworm
#from multiworm.readers import blob as blob_reader
#from settings.local import LOGISTICS
import waldo.conf.settings as settings


SAVE_DIR = os.path.abspath('../results/fig1-2014-07-31')

MWT_DIR = os.path.abspath(settings.LOGISTICS['filesystem_data'])
#VALIDATION_DIR = os.path.abspath(LOGISTICS['validation'])

def calculate_stats_for_moves(min_moves, prep_data):
    move = prep_data.load('moved')

    base_accuracy = prep_data.load('accuracy', index_col=False)
    print base_accuracy.head()
    print base_accuracy.columns
    matches = prep_data.load('matches')
    counts, tps, fps, fns = [], [], [], []
    for mm in min_moves:
        print mm
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

def calculate_duration_data(prep_data, min_move=2):
    move = prep_data.load('moved')
    terminals = prep_data.load('terminals')[['bid', 't0', 'tN']]
    terminals.set_index('bid', inplace=True)
    terminals.dropna(inplace=True)
    #print terminals.head()

    moved_bids = set(move[move['bl_moved'] >= min_move]['bid'])
    term_bids = set(terminals.index)
    term_overlap = term_bids & moved_bids
    missing_bids = moved_bids - term_overlap
    print(len(missing_bids), 'moved bids have no terminals')


    steps = terminals.loc[list(moved_bids)]
    steps = steps / 60.0
    steps.sort('t0', inplace=True)
    steps['lifespan'] = steps['tN'] - steps['t0']
    durations = np.array(steps['lifespan'])

    print(len(terminals), 'terminals')
    print(len(moved_bids), 'moved')
    print(len(steps), 'steps')

    return steps, durations

def step_plot(ax, step_df, color=1, label='moved > 2 BL', ylim=None):
    steps = []
    n_steps = len(step_df)

    xs = list(step_df['t0'])
    widths = list(step_df['lifespan'])
    height = 1

    color_cycle = ax._get_lines.color_cycle
    for i in range(color):
        color = color_cycle.next()
    #color2 = color_cycle.next()
    for y, (x, width) in enumerate(zip(xs, widths)):
        steps.append(patches.Rectangle((x,y), height=height, width=width,
                                       fill=True, fc=color, ec=color))
    for step in steps:
        ax.add_patch(step)

    xmax = 60
    ax.plot([0], color=color, label=label)
    ax.set_xlim([0, xmax])
    if ylim is None:
        ylim = n_steps + 1
    ax.set_ylim([0, ylim])

def ideal_step_plot(ax, n, xmax=60):
    n_steps = int(n)
    color_cycle = ax._get_lines.color_cycle
    color1 = color_cycle.next()
    color2 = color_cycle.next()

    xmax = 60
    for y in range(n_steps):
        ideal = patches.Rectangle((0,y), height=1, width=xmax,
                              fill=True, ec='black', fc=color2,
                              alpha=0.5)
        ax.add_patch(ideal)
    ax.plot([0], color=color2, alpha=0.5, label='ideal')
    ax.set_xlim([0, xmax])
    ax.set_ylim([0, n_steps+1])


def make_fig1(ex_id, save_name=None):

    experiment = wio.Experiment(experiment_id=ex_id)
    #prep_data = wio.file_manager.PrepData(ex_id)
    prep_data = experiment.prepdata

    # for panel 1
    #min_moves = [0, 0.25, 0.5, 1, 2]
    #data, counts = calculate_stats_for_moves(min_moves, prep_data)
    steps, durations = calculate_duration_data(prep_data, min_move=2)
    accuracy = prep_data.load('accuracy')
    worm_count = np.mean(accuracy['true-pos'] + accuracy['false-neg'])
    print 'worm count:', worm_count
    #estimated_count =
    #return steps
    #columns=min_moves).T
    #print data

    fig = plt.figure()
    # gs = gridspec.GridSpec(4, 7)
    # gs.update(left=0.05, wspace=1.0, hspace=1.0, right=0.95)
    # ax0 = plt.subplot(gs[:3,:5])
    # ax1 = plt.subplot(gs[0:2,5:7])
    # ax2 = plt.subplot(gs[2:,5:7])
    # ax3 = plt.subplot(gs[3,:5])
    fig.set_size_inches(30,10)
    gs = gridspec.GridSpec(5, 4)
    gs.update(left=0.1, wspace=1.0, hspace=0.1, right=0.9)
    ax0 = plt.subplot(gs[:3,:])
    ax1 = plt.subplot(gs[3:,:2])
    ax2 = plt.subplot(gs[3:,2:])

    ### AX 0
    color_cycle = ax0._get_lines.color_cycle

    c = {'tp_color': color_cycle.next(),
         'missed_color': color_cycle.next(),
         'fp_color': color_cycle.next(),
         'roi_color': color_cycle.next(),
         'roi_line_color': color_cycle.next()}
    wf.draw_colors_on_image_T(ex_id, time=30*30, ax=ax0, colors=c)

    ### AX 1
    step_plot(ax1, steps, color=1)
    ax1.set_ylabel('track number')
    ax1.set_xlabel('existance (min)')
    ax1.legend(loc='upper left')

    ### AX 2
    sim_steps = ssim.run_ideal_step_simulation(experiment)
    ylim = len(steps)
    step_plot(ax2, sim_steps, color=2, label='simulated ideal', ylim=ylim)
    ax2.set_ylabel('track number')
    ax2.set_xlabel('existance (min)')
    ax2.legend(loc='upper left')
    #plt.tight_layout()

    if save_name is None:
        plt.show()
    else:
        plt.savefig(fig_name, format='pdf')

def make_fig1_old(ex_id, save_name=None):

    prep_data = wio.file_manager.PrepData(ex_id)

    # for panel 1
    min_moves = [0, 0.25, 0.5, 1, 2]
    data, counts = calculate_stats_for_moves(min_moves, prep_data)
    steps, durations = calculate_duration_data(prep_data, min_move=2)
    #return steps
    #columns=min_moves).T
    #print data

    fig = plt.figure()
    gs = gridspec.GridSpec(4, 7)
    gs.update(left=0.05, wspace=1.0, hspace=1.0, right=0.95)
    ax0 = plt.subplot(gs[:,:3])
    ax1 = plt.subplot(gs[0:2,3:5])
    ax2 = plt.subplot(gs[0:2,5:])
    ax3 = plt.subplot(gs[2:,3:5])
    ax4 = plt.subplot(gs[2:,5:])

    ### AX 0
    color_cycle = ax0._get_lines.color_cycle

    c = {'tp_color':color_cycle.next(),
         'missed_color': color_cycle.next(),
         'fp_color': color_cycle.next(),
         'roi_color': color_cycle.next(),
         'roi_line_color': color_cycle.next()}
    wf.draw_colors_on_image(ex_id, time=30*60, ax=ax0, colors=c)

    ### AX 1
    min_moves = [0, 0.25, 0.5, 1, 2]
    for mm in min_moves:
        steps, durations = calculate_duration_data(prep_data, min_move=mm)
        ecdf = ECDF(durations)
        cdf = ecdf(range(65))
        ax1.plot(cdf, label='move > {m} BL'.format(m=mm))
    ax1.plot([0, 60, 60,65], [0,0, 1,1], label='ideal')
    ax1.legend(loc='lower center')
    ax1.set_ylim([0, 1.001])
    ax1.set_ylabel('CDF')
    ax1.set_xlabel('track duration (min)')

    ### AX 2
    counts.plot(kind='bar', ax=ax2)
    ax2.set_ylabel('# tracks')
    ax2.set_xticklabels([str(round(i,2)) for i in (min_moves)])
    ax2.set_xlabel('min. body-lengths moved')

    ### AX 3
    data.plot(kind='bar', ax=ax3, grid=True)
    ax3.set_ylabel('percent of true count (TP + FN)')
    ax3.set_ylim([0,100])
    ax3.set_xticklabels([str(round(i,2)) for i in (min_moves)])
    ax3.set_xlabel('min body-lengths moved')
    #ax3.legend(loc='lower right')

    ### AX 4

    step_plot(ax4, steps)
    ax4.set_ylabel('track number')
    ax4.set_xlabel('existance (min)')
    ax4.legend(loc='upper left')
    #plt.tight_layout()

    if save_name is None:
        plt.show()
    else:
        plt.savefig(fig_name, format='pdf')

ex_ids = ['20130614_120518',
          '20130318_131111',
          '20130414_140704', # giant component(?)
          '20130702_135704', # many pics
          '20130702_135652']

# for ex_id in ex_ids:
#     fig_name = '{d}/{eid}-fig1.pdf'.format(d=SAVE_DIR, eid=ex_id)
#     print fig_name
#     fm.ensure_dir_exists(SAVE_DIR)
#     try:
#         make_fig1(ex_id, save_name=fig_name)
#     except Exception as e:
#         print e

ex_id = '20130414_140704'
ex_id = '20130614_120518'
ex_id = '20130318_131111'
ex_id = '20130702_135704' # testset
#ex_id = '20130702_135652'

fig_name = '{d}/{eid}-fig1.pdf'.format(d=SAVE_DIR, eid=ex_id)
fig_name = None
s = make_fig1(ex_id, save_name=fig_name)
