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


class WaldoProcessDialog(QtGui.QDialog):
    def __init__(self, ex_id, func, finish_func, parent=None):
        super(WaldoProcessDialog, self).__init__(parent)
        self.finish_func = finish_func

        label = QtGui.QLabel("Running experiment: {ex_id}".format(ex_id=ex_id))
        progress_bar = QtGui.QProgressBar()
        progress_bar.setRange(0, 100)

        cancel_run_button = QtGui.QPushButton("Cancel")
        cancel_run_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cancel_run_button.clicked.connect(self.cancel_run_button_clicked)

        progress_layout = QtGui.QHBoxLayout()
        progress_layout.addWidget(progress_bar)
        progress_layout.addWidget(cancel_run_button)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addLayout(progress_layout)
        self.setLayout(layout)

        self.progress_bar = progress_bar
        self.cancel_run_button = cancel_run_button

        self.task = tasking.CommandTask(self.madeProgress, self.finished, self.cancelled)
        self.task.start(func)
        self.setFixedSize(self.minimumSize())
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint)

    def cancel_run_button_clicked(self):
        self.cancel_run_button.setEnabled(False)
        if self.task is not None:
            self.task.requestCancel()
        return False

    def madeProgress(self, item, value):
        if self.task is not None:
            self.progress_bar.setValue(value * 100)

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

    def waldoProcess(self, callback):
        STEPS = 5.0
        ex_id = self.data.ex_id
        callback(0, 0.0 / STEPS)
        prepare_summarize(ex_id)
        callback(0, 1.0 / STEPS)
        images_summarize(ex_id)
        callback(0, 2.0 / STEPS)
        experiment = Experiment(experiment_id=ex_id)
        callback(0, 3.0 / STEPS)
        graph, report_df = iterative_solver(experiment, experiment.graph.copy())
        callback(0, 4.0 / STEPS)
        out_writer = OutputWriter(ex_id, graph=graph)
        out_writer.export()
        callback(0, 5.0 / STEPS)

    def finished(self):
        self.waldoProcessCompleted = True
        self.completeChanged.emit()

    def isComplete(self):
        return self.waldoProcessCompleted