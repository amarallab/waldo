from __future__ import absolute_import, print_function

# standard library
import os
import glob
import json
import threading

# third party
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

# project specific
from waldo.conf import settings
from waldo.gui import tasking
from waldo.wio import paths
from . import pages

def get_summary_data(experiment_name):
    files = glob.glob(settings.MWT_DATA_ROOT + "/" + experiment_name + "/*.summary")
    summary_name = ""
    duration = ""

    if len(files) != 1:
        return AsyncSummaryLoader.ROW_STATE_INVALID, summary_name, duration

    summary_name = os.path.splitext(os.path.basename(files[0]))[0]
    with open(files[0], "rt") as f:
        lines = f.readlines()
        if len(lines) == 0:
            return AsyncSummaryLoader.ROW_STATE_INVALID, summary_name, duration

        lastline = lines[-1]
        params = lastline.split(' ')
        if len(params) < 2:
            return AsyncSummaryLoader.ROW_STATE_INVALID, summary_name, duration

        duration = params[1]

    filename = settings.PROJECT_DATA_ROOT + "/" + experiment_name + "/waldo/" + experiment_name + "-report-card.csv"
    if os.path.isfile(filename):
        state = AsyncSummaryLoader.ROW_STATE_VALID_HAS_RESULTS
    else:
        state = AsyncSummaryLoader.ROW_STATE_VALID
    return state, summary_name, duration


# Load data from an experiment list
class AsyncSummaryLoader(QtCore.QThread):
    ROW_STATE_INVALID = -1
    ROW_STATE_NOT_LOADED = 0
    ROW_STATE_VALID = 1
    ROW_STATE_VALID_HAS_RESULTS = 2

    row_summary_changed = QtCore.pyqtSignal([int, int, str, str])

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.lock = threading.Lock()
        self.sleep = threading.Semaphore(1)
        self.queue = []

    def startListening(self):
        self.finish = False
        self.start()

    def run(self):
        while not self.finish:
            self.lock.acquire()
            if len(self.queue) == 0:
                self.lock.release()
                self.sleep.acquire()
            else:
                row, folder, item = self.queue.pop(0)
                self.lock.release()

                try:
                    row_state, summary_name, duration = get_summary_data(folder)
                except:
                    import traceback
                    traceback.print_exc()
                self.row_summary_changed.emit(row, row_state, summary_name, duration)

    def stopListening(self):
        self.finish = True
        with self.lock:
            self.queue = []
        self.sleep.release()
        self.wait()
        self.terminate()

    def clearRows(self):
        with self.lock:
            self.queue = []

    def addRow(self, row, folder, item):
        self.lock.acquire()
        if len(self.queue) == 0:
            self.sleep.release()
        self.queue.append((row, folder, item))
        self.lock.release()


# Load threshold data from an experiment
class CacheThresholdLoadingDialog(QtGui.QDialog):
    def __init__(self, ex_id, func, finish_func, parent=None):
        super(CacheThresholdLoadingDialog, self).__init__(parent)
        self.finish_func = finish_func

        label = QtGui.QLabel("Loading experiment: {ex_id}".format(ex_id=ex_id))
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
        self.cancel_run_button.setText("Canceling")
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

