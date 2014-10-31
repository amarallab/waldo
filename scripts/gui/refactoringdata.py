__author__ = 'heltena'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

import threading
import time
import silly_test
import tasking

class RefactoringDataRunningDialog(QtGui.QDialog):
    def __init__(self, data, parent=None):
        super(RefactoringDataRunningDialog, self).__init__(parent)
        self.data = data

        TASK_NUM = 5
        main_progress_bar = QtGui.QProgressBar()
        main_progress_bar.setRange(0, TASK_NUM)

        subtask_progress_bar = QtGui.QProgressBar()
        subtask_progress_bar.setRange(0, 100)

        cancel_run_button = QtGui.QPushButton("Cancel")
        cancel_run_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cancel_run_button.clicked.connect(self.cancel_run_button_clicked)

        progress_layout = QtGui.QVBoxLayout()
        progress_layout.addWidget(main_progress_bar)
        progress_layout.addWidget(subtask_progress_bar)
        progress_layout.addWidget(cancel_run_button)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(progress_layout)
        self.setLayout(layout)

        self.main_progress_bar = main_progress_bar
        self.subtask_progress_bar = subtask_progress_bar
        self.cancel_run_button = cancel_run_button

        self.task = tasking.CommandTask(self.madeProgress, self.finished, self.cancelled)
        self.task.start(silly_test.silly_function_2, task_num=TASK_NUM, elapse=1)
        self.setFixedSize(self.minimumSize())
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint)

    def cancel_run_button_clicked(self):
        self.cancel_run_button.setEnabled(False)
        if self.task is not None:
            self.task.requestCancel()
        return False

    def madeProgress(self, item, value):
        if self.task is not None:
            if item == 0:
                self.main_progress_bar.setValue(value)
            elif item == 1:
                self.subtask_progress_bar.setValue(value * 100)

    def finished(self):
        self.task.waitFinished()
        self.task = None
        self.result = "Finished"
        self.close()

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


class RefactoringDataPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(RefactoringDataPage, self).__init__(parent)
        self.data = data
        self.setTitle("Score")
        self.setSubTitle("Press 'start' to start refactoring data.")

        start_button = QtGui.QPushButton("Start")
        start_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        start_button.clicked.connect(self.start_button_clicked)

        result_label = QtGui.QLabel("")
        result_label.setWordWrap(True)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(start_button)
        layout.addWidget(result_label)
        self.setLayout(layout)

        self.start_button = start_button
        self.result_label = result_label

    def initializePage(self):
        self.result_label.setText("")

    def start_button_clicked(self):
        dlg = RefactoringDataRunningDialog(self)
        dlg.setModal(True)
        dlg.exec_()
        self.result_label.setText(dlg.result)
