__author__ = 'heltena'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt, QTimer

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

from waldo.gui import pathcustomize
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
    def __init__(self, experiment, func, image_func, finish_func, parent=None):
        super(WaldoProcessDialog, self).__init__(parent)
        self.image_func = image_func
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
        main_layout.addWidget(QtGui.QLabel("Running experiment: {ex_id}".format(ex_id=experiment.id)), 0, 0, 1, 2)

        row = 1
        for label, progress_bar in zip(progress_bar_labels, progress_bars):
            main_layout.addWidget(QtGui.QLabel(label), row, 0, 1, 1)
            main_layout.addWidget(progress_bar, row, 1, 1, 1)
            row += 1

        main_layout.addWidget(cancel_run_button, row, 0, 1, 2)

        self.setLayout(main_layout)

        self.progress_bars = progress_bars
        self.cancel_run_button = cancel_run_button

        self.task = tasking.CommandTask(self.madeProgress, self.finished, self.cancelled, self.imageMadeProgress)
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
            if item < 10:
                self.progress_bars[item].setValue(value * 100)

    def imageMadeProgress(self, item, value):
        if self.task is not None:
            self.image_func(item, value)

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
        self.setTitle("Running WALDO")

        self.image_figure = plt.figure()
        self.image_canvas = FigureCanvas(self.image_figure)
        self.image_canvas.setMinimumSize(50, 50)
        self.image_canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.ax_image = self.image_figure.add_subplot(111)
        self.ax_image.axis('off')

        self.main_layout = QtGui.QHBoxLayout()
        self.showing_image = False

        layout = QtGui.QVBoxLayout()
        layout.addLayout(self.main_layout)
        self.setLayout(layout)

    def initializePage(self):
        self.waldoProcessCompleted = False

        QTimer.singleShot(0, self.show_dialog)

    def show_dialog(self):
        dlg = WaldoProcessDialog(self.data.experiment, self.waldoProcess, self.show_image, self.finished, self)
        dlg.setModal(True)
        dlg.exec_()

    def _set_image(self, image):
        if not self.showing_image:
            self.main_layout.addWidget(self.image_canvas)
            self.showing_image = True

        self.ax_image.clear()
        self.ax_image.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        self.ax_image.axis('off')

    def waldoProcess(self, callback):
        times, impaths = zip(*sorted(self.data.experiment.image_files.items()))
        impaths = [str(s) for s in impaths]

        self.last_image_index = 0
        def callback_with_image(x):
            if len(impaths) == 0:
                return
            index = int(x * len(impaths))
            if index > len(impaths) - 1:
                index = len(impaths) - 1
            if index - self.last_image_index < 1:
                return
            self.last_image_index = index
            im = mpimg.imread(impaths[index])
            callback(10, im)

        def PROCESS_BLOBS_CALLBACK(x):
            callback(1, x)
            callback_with_image(x)

        PROCESS_IMAGES_CALLBACK = lambda x: callback(2, x)
        LOAD_EXPERIMENT_CALLBACK = lambda x: callback(3, x)
        CORRECT_ERROR_CALLBACK = lambda x: callback(4, x)
        WRITE_OUTPUT_CALLBACK = lambda x: callback(5, x)
        GENERATE_REPORT_CALLBACK = lambda x: callback(6, x)
        NEW_IMAGE_CALLBACK = lambda im: callback(11, im)

        STEPS = 5.0
        ex_id = self.data.experiment.id
        callback(0, 0.0 / STEPS)

        prepare_summarize(ex_id, callback=PROCESS_BLOBS_CALLBACK)
        PROCESS_BLOBS_CALLBACK(1)
        callback(0, 1.0 / STEPS)

        images_summarize(ex_id, callback=PROCESS_IMAGES_CALLBACK, image_callback=NEW_IMAGE_CALLBACK)
        PROCESS_IMAGES_CALLBACK(1)
        callback(0, 2.0 / STEPS)

        experiment = self.data.experiment
        LOAD_EXPERIMENT_CALLBACK(1)
        callback(0, 3.0 / STEPS)

        graph = experiment.graph.copy()
        solver = WaldoSolver(experiment, graph)
        graph, report_df = solver.run(callback=CORRECT_ERROR_CALLBACK)
        CORRECT_ERROR_CALLBACK(1)
        callback(0, 4.0 / STEPS)

        out_writer = OutputWriter(experiment.id, graph=graph)
        out_writer.export(callback1=WRITE_OUTPUT_CALLBACK, callback2=GENERATE_REPORT_CALLBACK)
        WRITE_OUTPUT_CALLBACK(1)
        GENERATE_REPORT_CALLBACK(1)
        callback(0, 5.0 / STEPS)

        self.export_tables()

    def show_image(self, id, image):
        self._set_image(image)

    def export_tables(self):
        ex_id = self.data.experiment.id
        path = os.path.join(settings.PROJECT_DATA_ROOT, ex_id)
        report_card_df = self.data.experiment.prepdata.load('report-card')
        if report_card_df is not None:
            headers = ['phase', 'step', 'total-nodes', '>10min', '>20min', '>30min', '>40min', '>50min', 'duration-mean', 'duration-std']
            b = report_card_df[headers]
            name = os.path.join(path, ex_id + '-track_counts.csv')
            b.to_csv(path_or_buf=name)

            headers = ['phase', 'step', 'total-nodes', 'connected-nodes', 'isolated-nodes', 'giant-component-size', '# components']
            b = report_card_df[headers]
            name = os.path.join(path, ex_id + '-network_overview.csv')
            b.to_csv(path_or_buf=name)

        df = self.data.experiment.prepdata.load('end_report')
        if df is not None:
            df['lifespan'] = ['0-1 min', '1-5 min', '6-10 min', '11-20 min', '21-60 min', 'total']
            df = df.rename(columns={'lifespan': 'track-duration',
                                    'unknown': 'disappear',
                                    'timing':'recording-finishes',
                                    'on_edge':'image-edge'})
            df.set_index('track-duration', inplace=True)
            name = os.path.join(path, ex_id + '-tract_termination.csv')
            df.to_csv(path_or_buf=name)

        df = self.data.experiment.prepdata.load('start_report')
        if df is not None:
            df['lifespan'] = ['0-1 min', '1-5 min', '6-10 min', '11-20 min', '21-60 min', 'total']
            df = df.rename(columns={'lifespan': 'track-duration',
                                    'unknown': 'appear',
                                    'timing':'recording-begins',
                                    'on_edge':'image-edge'})
            df.set_index('track-duration', inplace=True)
            name = os.path.join(path, ex_id + '-tract_initiation.csv')
            df.to_csv(path_or_buf=name)

    def finished(self):
        self.waldoProcessCompleted = True
        self.completeChanged.emit()
        self.wizard().next()

    def isComplete(self):
        return self.waldoProcessCompleted
