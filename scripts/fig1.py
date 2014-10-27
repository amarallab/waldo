import pathcustomize
import os

import numpy as np

#from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpltools import style


#import sys
#sys.path.append('../scripts')
#print sys.path
#import pathcustomize

from waldo import wio
import waldo.images.evaluate_acuracy as ea
import waldo.images.worm_finder as wf
import waldo.metrics.report_card as report_card
import waldo.metrics.step_simulation as ssim
import waldo.viz.eye_plots as ep

SAVE_DIR = os.path.abspath('../results/fig1-2014-07-31')
style.use('ggplot')

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

def make_fig1(ex_id, step_min_move = 1, save_name=None):

    experiment = wio.Experiment(experiment_id=ex_id)
    prep_data = experiment.prepdata
    graph = experiment.graph.copy()
    # for panel 1
    #steps, durations = report_card.calculate_duration_data(experiment.prepdata, min_move=1)
    moving_nodes =  [int(i) for i in graph.compound_bl_filter(experiment, threshold=step_min_move)]
    steps, durations = report_card.calculate_duration_data_from_graph(experiment, graph, moving_nodes)
    accuracy = experiment.prepdata.load('accuracy')
    worm_count = np.mean(accuracy['true-pos'] + accuracy['false-neg'])
    print 'worm count:', worm_count

    fig = plt.figure()
    fig.set_size_inches(30,10)
    gs = gridspec.GridSpec(5, 4)
    gs.update(left=0.1, wspace=1.0, hspace=0.1, right=0.9)
    ax0 = plt.subplot(gs[:3,:])
    ax1 = plt.subplot(gs[3:,:2])
    ax2 = plt.subplot(gs[3:,2:])

    ### AX 0
    print 'starting ax 0'
    color_cycle = ax0._get_lines.color_cycle

    c = {'tp_color': color_cycle.next(),
         'missed_color': color_cycle.next(),
         'fp_color': color_cycle.next(),
         'roi_color': color_cycle.next(),
         'roi_line_color': color_cycle.next()}
    wf.draw_colors_on_image_T(ex_id, time=30*30, ax=ax0, colors=c)

    sim_steps = ssim.run_ideal_step_simulation(experiment)

    short_lim = 5
    step_kwargs = {'short_lim':short_lim, 'front_back_margin':10}

    ### AX 1
    print 'starting ax 1'
    ymax = len(steps[steps['lifespan'] > short_lim])
    legend_props = ep.eye_plot(ax1, steps, color=1, label='crawled > {bl} BL'.format(bl=step_min_move),
                            ymax=ymax, **step_kwargs)
    ax1.set_ylabel('track number')
    ax1.set_xlabel('existence (min)')
    ep.add_simulation_lines(ax1, sim_steps, ymax=ymax)
    ax1.legend(**legend_props)

    ### AX 2
    print 'starting ax 2'
    ymax = len(sim_steps[sim_steps['lifespan'] > short_lim])
    legend_props = ep.eye_plot(ax2, sim_steps, color=2, label='simulated ideal',
                            ymax=ymax, **step_kwargs)
    ep.add_simulation_lines(ax2, sim_steps, ymax=ymax)
    ax2.set_ylabel('track number')
    ax2.set_xlabel('existence (min)')
    ax2.legend(**legend_props)

    if save_name is None:
        plt.show()
    else:
        plt.savefig(fig_name, format='pdf')

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
#     ax4.set_xlabel('existence (min)')
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

    #ex_id = '20141017_134720'
    #ex_id = '20141017_134724'

    fig_name = '{d}/{eid}-fig1.pdf'.format(d=SAVE_DIR, eid=ex_id)
    fig_name = None
    s = make_fig1(ex_id, save_name=fig_name)
