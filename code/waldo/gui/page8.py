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
    def __init__(self, experiment, func, finish_func, parent=None):
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

        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

    def initializePage(self):
        self.waldoProcessCompleted = False

        dlg = WaldoProcessDialog(self.data.experiment, self.waldoProcess, self.finished, self)
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
        ex_id = self.data.experiment.id
        callback(0, 0.0 / STEPS)

        prepare_summarize(ex_id, callback=PROCESS_BLOBS_CALLBACK)
        PROCESS_BLOBS_CALLBACK(1)
        callback(0, 1.0 / STEPS)

        images_summarize(ex_id, callback=PROCESS_IMAGES_CALLBACK)
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

    def finished(self):
        self.waldoProcessCompleted = True
        self.completeChanged.emit()

    def isComplete(self):
        return self.waldoProcessCompleted
