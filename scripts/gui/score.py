__author__ = 'heltena'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

import threading
import time
import silly_test
import tasking

class ScorePage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(ScorePage, self).__init__(parent)
        self.data = data
        self.setTitle("Score")
        self.setSubTitle("Press 'start' to start scoring.")

        progressBar = QtGui.QProgressBar()
        progressBar.setRange(0, 100)
        progressBar.setValue(0)

        cancelButton = QtGui.QPushButton("Cancel")
        cancelButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cancelButton.clicked.connect(self.cancelButton_clicked)

        progressLayout = QtGui.QHBoxLayout()
        progressLayout.addWidget(progressBar)
        progressLayout.addWidget(cancelButton)

        startButton = QtGui.QPushButton("Start")
        startButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        startButton.clicked.connect(self.startButton_clicked)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(progressLayout)
        layout.addWidget(startButton)
        self.setLayout(layout)

        self.progressBar = progressBar
        self.startButton = startButton
        self.cancelButton = cancelButton
        self.task = None

    def initializePage(self):
        self.progressBar.setValue(0)
        self.startButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
        self.scoreFinished = False
        self.task = None

    def cleanupPage(self):
        if self.task is not None:
            self.task.requestCancel()
            self.task = None

    def startButton_clicked(self):
        self.progressBar.setValue(0)
        self.startButton.setEnabled(False)
        self.cancelButton.setEnabled(True)
        self.scoreFinished = False
        self.completeChanged.emit()

        self.task = tasking.CommandTask(self.madeProgress, self.finished)
        self.task.start(silly_test.silly_function, elapse=1)

    def cancelButton_clicked(self):
        if self.task is not None:
            self.task.requestCancel()
            self.task = None
        self.progressBar.setValue(0)
        self.startButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
        self.scoreFinished = False
        self.completeChanged.emit()

    def madeProgress(self, value):
        if self.task is not None:
            self.progressBar.setValue(value * 100)

    def finished(self):
        if self.task is not None:
            self.task.requestCancel()
            self.task = None
        self.progressBar.setValue(100)
        self.scoreFinished = True
        self.startButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
        self.completeChanged.emit()

    def isComplete(self):
        return self.scoreFinished

