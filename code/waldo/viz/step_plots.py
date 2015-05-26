import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.patches as patches

from waldo import wio
#import waldo.metrics.report_card as report_card
# import waldo.images.evaluate_acuracy as ea
# import waldo.images.worm_finder as wf
# import waldo.metrics.step_simulation as ssim
# import waldo.viz.eye_plots as ep
# from waldo.gui import pathcustomize
plt.style.use('bmh')


class StepPlot(object):

    def __init__(self):
        self.plot = plt.figure(figsize=(7, 8))

    def _new_plot(self, ax, df, label, nth_color=0, xmax=60):
        step_df = df
        steps = []

        xs = list(step_df['t0'])
        widths = list(step_df['lifespan'])
        height = 1

        color = ax._get_lines.color_cycle.next()
        for i in range(nth_color):
            color = ax._get_lines.color_cycle.next()

        for y, (x, width) in enumerate(zip(xs, widths)):
            steps.append(patches.Rectangle((x, y), height=height, width=width,
                                           fill=True, fc=color, ec=color,
                                           alpha=0.5))
        for step in steps:
            ax.add_patch(step)

        ax.plot([0], color=color, label=label, alpha=0.5)
        ax.set_xlim([0, xmax])
        tick_label_size = 12
        plt.yticks(fontsize=tick_label_size)
        plt.xticks(fontsize=tick_label_size)

    def steps_from_node_report(self, experiment, min_bl=1):
        node_report = experiment.prepdata.load('node-summary')
        # print(node_report.head())
        steps = node_report[['bl', 't0', 'tN', 'bid']]
        steps.set_index('bid', inplace=True)

        steps.loc[:, 't0'] = steps['t0'] / 60.0 
        steps.loc[:, 'tN'] = steps['tN'] / 60.0
        steps.loc[:, 'lifespan'] = steps['tN'] - steps['t0']
        steps.loc[:, 'mid'] = (steps['tN'] + steps['t0']) / 2.0
        return steps[steps['bl'] >= 1]

    def make_figures(self, ex_id, step_min_move=1, rescale_to_hours=True):
        experiment = wio.Experiment(experiment_id=ex_id)
        graph = experiment.graph.copy()

        self.plot.clf()

        moving_nodes = [int(i) for i
                        in graph.compound_bl_filter(experiment,
                                                    threshold=step_min_move)]
        steps, durations = report_card.calculate_duration_data_from_graph(experiment, graph, moving_nodes)

        final_steps = self.steps_from_node_report(experiment)
        accuracy = experiment.prepdata.load('accuracy')
        worm_count = np.mean(accuracy['true-pos'] + accuracy['false-neg'])
        if rescale_to_hours:
            steps = steps / 60.0
            final_steps = final_steps / 60.0
        # print('worm count:', worm_count)
        #print(steps)
        ### AX 0
        # print('starting ax 0')
        # color_cycle = ax._get_lines.color_cycle
        # for i in range(5):
        #     c = color_cycle.next()
        # wf.draw_minimal_colors_on_image_T(ex_id, time=30*30, ax=ax, color=c)
        self.step_facets(steps, final_steps)

    def step_facets(self, df, df2, t1=5, t2=20, rescale_to_hours=True):
        xmax = max([max(df['tN']), max(df['tN'])])
        label_size = 16
        tick_label_size = 12
        gs = grd.GridSpec(3, 2)
        gs.update(wspace=0.01, hspace=0.05)

        plot = self.plot
        ax_l0 = plot.add_subplot(gs[0, 0], axisbg='white')
        ax_l1 = plot.add_subplot(gs[0, 1], sharey=ax_l0, axisbg='white')
        ax_m0 = plot.add_subplot(gs[1, 0], axisbg='white')
        ax_m1 = plot.add_subplot(gs[1, 1], sharey=ax_m0, axisbg='white')
        ax_s0 = plot.add_subplot(gs[2, 0], axisbg='white')
        ax_s1 = plot.add_subplot(gs[2, 1], sharey=ax_s0, axisbg='white')

        left = [ax_l0, ax_m0, ax_s0]
        right = [ax_l1, ax_m1, ax_s1]
        top = [ax_l0, ax_l1]
        mid = [ax_m0, ax_m1]
        bottom = [ax_s0, ax_s1]
        all_axes = [ax_l0, ax_m0, ax_s0, ax_l1, ax_m1, ax_s1]
        
        ylabels = ['> {t2} min'.format(t2=t2),
                   '{t1} to {t2} min'.format(t1=t1, t2=t2),
                   '< {t1} min'.format(t1=t1)]

        if rescale_to_hours:
            t1 = t1 / 60.0
            t2 = t2 / 60.0

        short1 = df[df['lifespan'] < t1]
        short2 = df2[df2['lifespan'] < t1]
        mid1 = df[(df['lifespan'] < t2) & (df['lifespan'] >= t1)]
        mid2 = df2[(df2['lifespan'] < t2) & (df2['lifespan'] >= t1)]
        long1 = df[df['lifespan'] >= t2]
        long2 = df2[df2['lifespan'] >= t2]

        short_max = max([len(short1), len(short2)])
        mid_max = max([len(mid1), len(mid2)])
        long_max = max([len(long1), len(long2)])

        self._new_plot(ax_s0, short1, 'raw', xmax=xmax)
        self._new_plot(ax_m0, mid1, 'raw', xmax=xmax)
        self._new_plot(ax_l0, long1, 'raw', xmax=xmax)

        self._new_plot(ax_s1, short2, 'final', nth_color=1, xmax=xmax)
        self._new_plot(ax_m1, mid2, 'final', nth_color=1, xmax=xmax)
        self._new_plot(ax_l1, long2, 'final', nth_color=1, xmax=xmax)

        for ax in right:
            ax.axes.yaxis.tick_right()

        for ax, t in zip(top, ['raw', 'final']):
            ax.get_xaxis().set_ticklabels([])
            ax.set_ylim([0, long_max])

        for ax in mid:
            ax.get_xaxis().set_ticklabels([])
            ax.set_ylim([0, mid_max])

        for ax in bottom:
            ax.set_xlabel('time (hours)', size=label_size)
            ax.set_ylim([0, short_max])

        # for ax, t in zip(left, ylabels):
        #     ax.set_ylabel(t, size=label_size)
            
        # for ax in all_axes:
        #     ax.get_xaxis()
        #     ax.get_yaxis()
        #     plt.yticks(fontsize=tick_label_size)
        #     plt.xticks(fontsize=tick_label_size)
            
        #tick_labels = ax_s0.xaxis.get_majorticklabels()
        ax_s0.get_xaxis().set_ticklabels([0, 10, 20, 30, 40, 50])
        ax_s0.get_xaxis().set_ticklabels([0, 2, 4, 6, 8, 10])