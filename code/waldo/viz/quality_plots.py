import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.patches as patches
import pathlib

class StepPlot(object):

    def __init__(self, experiment):
        self.plot = plt.figure(figsize=(7, 9))
        self.e = experiment

    def _new_plot(self, ax, df, label, nth_color=0, xmax=60, alpha=0.5):
        step_df = df
        steps = []

        xs = list(step_df['t0'])
        widths = list(step_df['lifespan'])
        height = 0.9 

        color = 'steelblue'

        for y, (x, width) in enumerate(zip(xs, widths)):
            steps.append(patches.Rectangle((x, y), height=height, width=width,
                                           fill=True, fc=color, ec=color,
                                           alpha=alpha))
        for step in steps:
            ax.add_patch(step)

        x = 5.0/6.0
        y = y + 1 
        print(x, y)
        ax.plot([x, x], [0, y], color='darkred', label=label, alpha=0.5)
        ax.plot([0], color=color, label=label, alpha=0.5)
        ax.set_xlim([0, xmax])
        ax.set_ylim([0, y])


    # def steps_from_node_report(self, fullpath, min_bl=1):
    #     print(fullpath)
    #     w_file_path = fullpath / '..' / 'waldo'
    #     ns_file = None
    #     for i in w_file_path.glob('*.csv'):
    #         if 'node-summary' in i.name:
    #             ns_file = i
    #             break

    #     if ns_file is None:
    #         print('WARNING. no node-report file found in {p}'.format(p=w_file_path))
    #         return None

    #     node_report = pd.read_csv(str(ns_file))

    #     steps = node_report[['bl', 't0', 'tN', 'bid']].copy()
    #     steps.set_index('bid', inplace=True)

    #     steps.loc[:, 't0'] = steps['t0'] / 60.0
    #     steps.loc[:, 'tN'] = steps['tN'] / 60.0
    #     steps.loc[:, 'lifespan'] = steps['tN'] - steps['t0']
    #     steps.loc[:, 'mid'] = (steps['tN'] + steps['t0']) / 2.0
    #     return steps[steps['bl'] >= 1]

    def steps_from_node_report(self, min_bl=1):
        experiment = self.e
        node_report = experiment.prepdata.load('node-summary')
        # print(node_report.head())
        steps = node_report[['bl', 't0', 'tN', 'bid']]
        steps.set_index('bid', inplace=True)

        steps.loc[:, 't0'] = steps['t0'] / 60.0
        steps.loc[:, 'tN'] = steps['tN'] / 60.0
        steps.loc[:, 'lifespan'] = steps['tN'] - steps['t0']
        steps.loc[:, 'mid'] = (steps['tN'] + steps['t0']) / 2.0
        return steps[steps['bl'] >= 1]

    def make_figures(self, step_min_move=1, rescale_to_hours=True, ax1=None, ax2=None):
        e = self.e 

        final_steps = self.steps_from_node_report() #fullpath=e.directory)
        if final_steps is None:
            return
        if rescale_to_hours:
            final_steps = final_steps / 60.0
        
        df2=final_steps
        t1=5
        t2=40
        rescale_to_hours=True
        
        xmax = max(df2['tN'])
        label_size = 16
        tick_label_size = 12

        tick_label_size = 20

        ax_l0 = ax1 #plot.add_subplot(gs[1, 0], axisbg='white')
        # plt.yticks([0, 10, 20], [], fontsize=tick_label_size)
        # plt.xticks([0, 4, 8, 12], [], fontsize=tick_label_size)
        # ax_l0.set_ylim([0, 20])

        ax_m0 = ax2 #plot.add_subplot(gs[0, 0], axisbg='white')
        ax_m0.xaxis.tick_top()

        # left = [ax_l0, ax_m0, ax_s0]
        top = [ax_l0] #, ax_l1]
        mid = [ax_m0] #, ax_m1]
        # bottom = [ax_s0] #, ax_s1]

        if rescale_to_hours:
            t1 = t1 / 60.0
            t2 = t2 / 60.0

        short2 = df2[df2['lifespan'] < t1]
        mid2 = df2[(df2['lifespan'] < t2) & (df2['lifespan'] >= t1)]
        long2 = df2[df2['lifespan'] >= t2]

        short_max = len(short2)
        mid_max = len(mid2)
        long_max = len(long2)

        # self._new_plot(ax_s0, short2, 'final', nth_color=0, xmax=xmax)
        self._new_plot(ax_m0, mid2, 'final', nth_color=0, xmax=xmax)
        self._new_plot(ax_l0, long2, 'final', nth_color=0, xmax=xmax)
        ax_l0.set_xlabel('time (hrs)')
        ax_m0.set_ylabel('ordered tracks')
        ax_l0.set_ylabel('ordered tracks')
                    
def squiggle_plot(e, ax):
    mins = []
    maxs = []
    lens = []

    for (bid,blob) in e.blobs():
        df = blob.df
        if df is not None:
            if len(df) > 50:
                x, y = zip(*df.centroid[::10])
                mins.append((min(x), min(y)))
                maxs.append((max(x), max(y)))
                lens.append(len(df))
                ax.plot(x, y, '.', markersize=0.3)
        
    mins = np.array(mins)
    maxs = np.array(maxs)
    mmax = maxs.max() + 100
    mmin = mins.min() - 100
    lims = [mmin, mmax]

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_yticks([], [])
    ax.set_xticks([], [])

def quality_control_plot(eid, experiment, plot_dir):
    fig = plt.figure(figsize=(16, 10), dpi=500)
    gs = grd.GridSpec(5, 8, wspace=1, hspace=1)

    step_top_ax = plt.subplot(gs[1:3, 0:4])
    step_bot_ax = plt.subplot(gs[3:5, 0:4])
    squiggle_ax = plt.subplot(gs[1:5, 4:8])

    for ax in [step_top_ax, step_bot_ax, squiggle_ax]:
        ax.set_axis_bgcolor('white')
        [i.set_linewidth(0.5) for i in ax.spines.itervalues()]

    # Set title
    title = '{eid}\n {name}'.format(eid=eid, name=experiment.basename)
    fig.suptitle(title, size=20)

    st = StepPlot(experiment=experiment)
    st.make_figures(ax2=step_top_ax, ax1=step_bot_ax)
    step_top_ax.text(2.9, 0.1, '5 < t < 40 min', fontsize=16,
                     verticalalignment='bottom', horizontalalignment='right',
                     )

    step_bot_ax.text(2.9, 0.1, 't > 40 min', fontsize=16,
                     verticalalignment='bottom', horizontalalignment='right',
                     )

    # Squiggle Plot!
    squiggle_plot(e=experiment, ax=squiggle_ax)
    gs.tight_layout(fig)



    path = pathlib.Path(plot_dir)
    if not path.is_dir():
        path.mkdir()
    name = path / '{eid}-check.png'.format(eid=eid) 
    fig.savefig(str(name))