__author__ = 'heltena'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

from waldo.conf import settings
from waldo.prepare import summarize as prepare_summarize
from waldo.images import summarize as images_summarize
from waldo.metrics.report_card import WaldoSolver
from waldo.wio import Experiment
from waldo.output.writer import OutputWriter

from waldo import wio
import waldo.images.evaluate_acuracy as ea
import waldo.images.worm_finder as wf
import waldo.metrics.report_card as report_card

import waldo.metrics.step_simulation as ssim
import waldo.viz.eye_plots as ep

import pathcustomize
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpltools import style

import matplotlib.image as mpimg
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from skimage import morphology
from skimage.measure import regionprops


style.use('ggplot')


import tasking

from time import time

class WaldoProcessDialog(QtGui.QDialog):
    def __init__(self, ex_id, func, finish_func, parent=None):
        super(WaldoProcessDialog, self).__init__(parent)
        self.finish_func = finish_func

        progress_bar_labels = ["Global progress", "Process blobs", "Process images", "Load experiment",
                               "Correct errors", "Write output", "Generate report"]
        progress_bars = []
        for i in range(len(progress_bar_labels)):
            progress_bar = QtGui.QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bars.append(progress_bar)

        cancel_run_button = QtGui.QPushButton("Cancel")
        cancel_run_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cancel_run_button.clicked.connect(self.cancel_run_button_clicked)

        main_layout = QtGui.QGridLayout()
        main_layout.addWidget(QtGui.QLabel("Running experiment: {ex_id}".format(ex_id=ex_id)), 0, 0, 1, 2)

        row = 1
        for label, progress_bar in zip(progress_bar_labels, progress_bars):
            main_layout.addWidget(QtGui.QLabel(label), row, 0, 1, 1)
            main_layout.addWidget(progress_bar, row, 1, 1, 1)
            row += 1

        main_layout.addWidget(cancel_run_button, row, 0, 1, 2)

        self.setLayout(main_layout)

        self.progress_bars = progress_bars
        self.cancel_run_button = cancel_run_button

        self.task = tasking.CommandTask(self.madeProgress, self.finished, self.cancelled)
        self.task.start(func)
        self.setFixedSize(self.minimumSize())
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint)

    def cancel_run_button_clicked(self):
        self.cancel_run_button.setEnabled(False)
        self.cancel_run_button.setText("Canceling")
        if self.task is not None:
            self.task.requestCancel()
        return False

    def madeProgress(self, item, value):
        if self.task is not None:
            self.progress_bars[item].setValue(value * 100)

    def finished(self):
        self.task.waitFinished()
        self.task = None
        self.result = "Finished"
        self.close()
        self.finish_func()

    def cancelled(self):
        self.task.waitFinished()
        self.task = None
        self.result = "Cancelled"
        self.close()

    def closeEvent(self, ev):
        if self.task is None:
            ev.accept()
        else:
            ev.ignore()


class WaldoProcessPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(WaldoProcessPage, self).__init__(parent)

        self.data = data
        self.waldoProcessCompleted = False
        self.setTitle("Waldo Process")

        self.tab = QtGui.QTabWidget()
        self.plot_titles = ['Figure 1', 'Figure 2', 'Figure 3']
        self.plots = []
        for title in self.plot_titles:
            figure = plt.figure()
            canvas = FigureCanvas(figure)
            canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
            canvas.setMinimumSize(50, 50)
            toolbar = NavigationToolbar(canvas, self)
            toolbar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

            widget = QtGui.QWidget()
            layout = QtGui.QVBoxLayout()
            layout.addWidget(canvas)
            layout.addWidget(toolbar)
            widget.setLayout(layout)

            self.tab.addTab(widget, title)
            self.plots.append(figure)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.tab)
        self.setLayout(layout)

    def initializePage(self):
        self.waldoProcessCompleted = False

        dlg = WaldoProcessDialog(self.data.ex_id, self.waldoProcess, self.finished, self)
        dlg.setModal(True)
        dlg.exec_()

    def waldoProcess(self, callback):
        PROCESS_BLOBS_CALLBACK = lambda x: callback(1, x)
        PROCESS_IMAGES_CALLBACK = lambda x: callback(2, x)
        LOAD_EXPERIMENT_CALLBACK = lambda x: callback(3, x)
        CORRECT_ERROR_CALLBACK = lambda x: callback(4, x)
        WRITE_OUTPUT_CALLBACK = lambda x: callback(5, x)
        GENERATE_REPORT_CALLBACK = lambda x: callback(6, x)

        STEPS = 5.0
        ex_id = self.data.ex_id
        callback(0, 0.0 / STEPS)

        prepare_summarize(ex_id, callback=PROCESS_BLOBS_CALLBACK)
        PROCESS_BLOBS_CALLBACK(1)
        callback(0, 1.0 / STEPS)

        images_summarize(ex_id, callback=PROCESS_IMAGES_CALLBACK)
        PROCESS_IMAGES_CALLBACK(1)
        callback(0, 2.0 / STEPS)

        experiment = Experiment(experiment_id=ex_id)
        LOAD_EXPERIMENT_CALLBACK(1)
        callback(0, 3.0 / STEPS)

        graph = experiment.graph.copy()
        solver = WaldoSolver(experiment, graph)
        graph, report_df = solver.run(callback=CORRECT_ERROR_CALLBACK)
        CORRECT_ERROR_CALLBACK(1)
        callback(0, 4.0 / STEPS)

        out_writer = OutputWriter(ex_id, graph=graph)
        out_writer.export(callback1=WRITE_OUTPUT_CALLBACK, callback2=GENERATE_REPORT_CALLBACK)
        WRITE_OUTPUT_CALLBACK(1)
        GENERATE_REPORT_CALLBACK(1)
        callback(0, 5.0 / STEPS)

    def finished(self):
        self.waldoProcessCompleted = True
        self.make_figures(self.data.ex_id)
        self.completeChanged.emit()

    def isComplete(self):
        return self.waldoProcessCompleted

    # MAKE FIG

    def calculate_stats_for_moves(self, min_moves, prep_data):
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

    def _new_plot(self, ax, df, label, nth_color=0, xmax=60):
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

    def steps_from_node_report(self, experiment, min_bl=1):
        node_report = experiment.prepdata.load('node-summary')
        print node_report.head()
        steps = node_report[['bl', 't0', 'tN', 'bid']]
        steps.set_index('bid', inplace=True)

        steps['t0'] = steps['t0'] / 60.0
        steps['tN'] = steps['tN'] / 60.0
        steps['lifespan'] = steps['tN'] - steps['t0']
        steps['mid'] = (steps['tN']  + steps['t0']) / 2.0
        return steps[steps['bl'] >= 1]

    def make_figures(self, ex_id, step_min_move = 1):
        experiment = wio.Experiment(experiment_id=ex_id)
        graph = experiment.graph.copy()

        moving_nodes =  [int(i) for i in graph.compound_bl_filter(experiment, threshold=step_min_move)]
        steps, durations = report_card.calculate_duration_data_from_graph(experiment, graph, moving_nodes)

        final_steps = self.steps_from_node_report(experiment)
        accuracy = experiment.prepdata.load('accuracy')
        worm_count = np.mean(accuracy['true-pos'] + accuracy['false-neg'])
        print 'worm count:', worm_count

        ### AX 0
        # print 'starting ax 0'
        # color_cycle = ax._get_lines.color_cycle
        # for i in range(5):
        #     c = color_cycle.next()
        # wf.draw_minimal_colors_on_image_T(ex_id, time=30*30, ax=ax, color=c)
        self.step_facets(steps, final_steps)

    def step_facets(self, df, df2, t1=5, t2=20):
        xmax = max([max(df['tN']), max(df['tN'])])

        gs = gridspec.GridSpec(3, 2)
        gs.update(wspace=0.01, hspace=0.05) #, bottom=0.1, top=0.7)

        ax_l0 = self.plots[0].add_subplot(gs[0,0])
        ax_l1 = self.plots[0].add_subplot(gs[0,1], sharey=ax_l0)
        ax_m0 = self.plots[0].add_subplot(gs[1,0])
        ax_m1 = self.plots[0].add_subplot(gs[1,1], sharey=ax_m0)
        ax_s0 = self.plots[0].add_subplot(gs[2,0])
        ax_s1 = self.plots[0].add_subplot(gs[2,1], sharey=ax_s0)

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
        print 'short', short_max
        print 'long', long_max
        print 'mid', mid_max
        print long1.head(20)

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
            ax.set_xlabel('t (min)')
            ax.set_ylim([0, short_max])

        for ax, t in zip(left, ylabels):
            ax.set_ylabel(t)

        ax_s0.get_xaxis().set_ticklabels([0, 10, 20, 30, 40, 50])
