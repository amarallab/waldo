__author__ = 'heltena'

import os

from PyQt4 import QtGui
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt, QTimer

from waldo.gui import tasking
import numpy as np
from scipy import ndimage
import json
import errno

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from skimage import morphology
from skimage.measure import regionprops

#from waldo.images.grab_images import grab_images_in_time_range
from waldo.conf import guisettings
from waldo import images

class ScoringDialog(QtGui.QDialog):
    def __init__(self, experiment, func, finish_func, parent=None):
        super(ScoringDialog, self).__init__(parent)
        self.finish_func = finish_func

        label = QtGui.QLabel("Scoring experiment: {ex_id}".format(ex_id=experiment.id))
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


class ScoringPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(ScoringPage, self).__init__(parent)

        self.data = data
        self.setTitle("Scoring")
        self.result = {}
        self.scoreCompleted = False

        self.contrastRatioResult = QtGui.QLabel()
        self.contrastDiffResult = QtGui.QLabel()
        self.goodFractionResult = QtGui.QLabel()
        self.coverageResult = QtGui.QLabel()

        layout = QtGui.QGridLayout()
        layout.addWidget(QtGui.QLabel("Contrast Ratio"), 0, 0, 1, 1)
        layout.addWidget(self.contrastRatioResult, 0, 1, 1, 1)

        layout.addWidget(QtGui.QLabel("Contrast Diff"), 1, 0, 1, 1)
        layout.addWidget(self.contrastDiffResult, 1, 1, 1, 1)

        layout.addWidget(QtGui.QLabel("Good Fraction"), 2, 0, 1, 1)
        layout.addWidget(self.goodFractionResult, 2, 1, 1, 1)

        layout.addWidget(QtGui.QLabel("Coverage"), 3, 0, 1, 1)
        layout.addWidget(self.coverageResult, 3, 1, 1, 1)

        self.setLayout(layout)

    def initializePage(self):
        self.contrastRatioResult.setText("")
        self.contrastDiffResult.setText("")
        self.goodFractionResult.setText("")
        self.coverageResult.setText("")
        self.scoreCompleted = False
        QTimer.singleShot(0, self.show_dialog)

    def show_dialog(self):
        dlg = ScoringDialog(self.data.experiment, self.scoring, self.finished, self)
        dlg.setModal(True)
        dlg.exec_()

    def scoring(self, callback):
        self.result = {}
        #try:
        self.result = images.score(None, experiment=self.data.experiment)
        #except:
        #    self.result = {}
        callback(0, 1)

    def _resultToString(self, value, range):
        if value is None:
            return "<b style='color: red'>Fail</b>", False
        if range[0] <= value < range[1]:
            return "<b style='color: green'>%f</b> (%f - %f)" % (value, range[0], range[1]), True
        else:
            return "<b style='color: red'>%f</b> (%f - %f)" % (value, range[0], range[1]), False

    def finished(self):
        valid = True

        text, completed = self._resultToString(
                self.result.get('contrast_ratio', None),
                guisettings.SCORE_CONTRAST_RADIO_RANGE)
        self.contrastRatioResult.setText(text)
        valid = valid and completed

        text, completed = self._resultToString(
                self.result.get('contrast_diff', None),
                guisettings.SCORE_CONTRAST_DIFF_RANGE)
        self.contrastDiffResult.setText(text)
        valid = valid and completed

        text, completed = self._resultToString(
                self.result.get('good_fraction', None),
                guisettings.SCORE_GOOD_FRACTION_RANGE)
        self.goodFractionResult.setText(text)
        valid = valid and completed

        text, completed = self._resultToString(
                self.result.get('coverage', None),
                guisettings.SCORE_COVERAGE_RANGE)
        self.coverageResult.setText(text)
        valid = valid and completed

        self.scoreCompleted = completed
        self.completeChanged.emit()

    def isComplete(self):
        return self.scoreCompleted
