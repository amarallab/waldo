__author__ = 'heltena'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

from waldo.conf import settings
from waldo.prepare import summarize as prepare_summarize
from waldo.images import summarize as images_summarize
from waldo.metrics.report_card import iterative_solver
from waldo.wio import Experiment
from waldo.output.writer import OutputWriter

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
        self.setSubTitle("TO-DO.")

        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

    def initializePage(self):
        self.waldoProcessCompleted = False

        dlg = WaldoProcessDialog(self.data.ex_id, self.waldoProcess, self.finished, self)
        dlg.setModal(True)
        dlg.exec_()

    def test(self, x, v, callback):
        print "TEST:", x
        return callback(v, x)

    def waldoProcess(self, callback):
        PROCESS_BLOBS_CALLBACK=lambda x: callback(1, x)
        PROCESS_IMAGES_CALLBACK=lambda x: callback(2, x)
        LOAD_EXPERIMENT_CALLBACK=lambda x: callback(3, x)
        CORRECT_ERROR_CALLBACK=lambda x: self.test(x, 4, callback)
        WRITE_OUTPUT_CALLBACK=lambda x: callback(5, x)
        GENERATE_REPORT_CALLBACK=lambda x: callback(6, x)

        STEPS = 5.0
        ex_id = self.data.ex_id
        callback(0, 0.0 / STEPS)

        #prepare_summarize(ex_id, callback=PROCESS_BLOBS_CALLBACK)
        PROCESS_BLOBS_CALLBACK(1)
        callback(0, 1.0 / STEPS)

        #images_summarize(ex_id, callback=PROCESS_IMAGES_CALLBACK)
        PROCESS_IMAGES_CALLBACK(1)
        callback(0, 2.0 / STEPS)

        experiment = Experiment(experiment_id=ex_id)
        LOAD_EXPERIMENT_CALLBACK(1)
        callback(0, 3.0 / STEPS)

        graph = experiment.graph.copy()
        graph, report_df = iterative_solver(experiment, graph, callback=CORRECT_ERROR_CALLBACK)
        CORRECT_ERROR_CALLBACK(1)
        callback(0, 4.0 / STEPS)

        out_writer = OutputWriter(ex_id, graph=graph)
        out_writer.export()
        callback(0, 5.0 / STEPS)

    def finished(self):
        self.waldoProcessCompleted = True
        self.completeChanged.emit()

    def isComplete(self):
        return self.waldoProcessCompleted