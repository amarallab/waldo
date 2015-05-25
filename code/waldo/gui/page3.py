from __future__ import absolute_import, print_function

__author__ = 'heltena'

# standard library
# import os
# import json

# third party
from PyQt4 import QtGui, QtCore
# from PyQt4.QtGui import QSizePolicy
# from PyQt4.QtCore import Qt

# project specific
# from waldo.wio import paths
from . import pages
from .helpers import experiment_has_thresholdCache, experiment_has_final_results


class PreviousThresholdCachePage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(PreviousThresholdCachePage, self).__init__(parent)

        self.data = data
        self.setTitle("Threshold Cache")
        self.setSubTitle("The next page will load the threshold cache data. It could take a few minutes.")

        self.recalculateThresholdButton = QtGui.QRadioButton("Choose Threshold and Calculate Final Results")
        self.skipThresholdButton = QtGui.QRadioButton("Calculate Final Result with Current Threshold")
        self.showFinalResultsButton = QtGui.QRadioButton("Show The Final Result")

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.recalculateThresholdButton)
        layout.addWidget(self.skipThresholdButton)
        layout.addWidget(self.showFinalResultsButton)
        self.setLayout(layout)

    def initializePage(self):
        self.data.loadSelectedExperiment()
        if self.data.experiment is None or not experiment_has_thresholdCache(self.data.experiment.id):
            self.recalculateThresholdButton.setVisible(False)
            self.skipThresholdButton.setVisible(False)
            self.showFinalResultsButton.setVisible(False)
        elif not experiment_has_final_results(self.data.experiment.id):
            self.skipThresholdButton.setChecked(True)
            self.recalculateThresholdButton.setVisible(True)
            self.skipThresholdButton.setVisible(True)
            self.showFinalResultsButton.setVisible(False)
        else:
            self.showFinalResultsButton.setChecked(True)
            self.recalculateThresholdButton.setVisible(True)
            self.skipThresholdButton.setVisible(True)
            self.showFinalResultsButton.setVisible(True)

    def nextId(self):
        if self.recalculateThresholdButton.isChecked():
            return pages.THRESHOLD_CACHE
        elif self.skipThresholdButton.isChecked():
            return pages.PREVIOUS_SCORING
        else:
            return pages.FINAL

