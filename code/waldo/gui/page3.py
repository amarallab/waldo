from __future__ import absolute_import, print_function

__author__ = 'heltena'

# standard library
import os
import json

# third party
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

# project specific
from waldo.wio import paths
from . import pages
from .helpers import experiment_has_thresholdCache


class PreviousThresholdCachePage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(PreviousThresholdCachePage, self).__init__(parent)

        self.data = data
        self.setTitle("Threshold Cache")
        self.setSubTitle("The next page will load the threshold cache data. It could take a few minutes.")

        self.recalculateDataCheckbox = QtGui.QCheckBox("Recalculate data.")
        self.recalculateDataCheckbox.setVisible(False)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.recalculateDataCheckbox)
        self.setLayout(layout)

    def initializePage(self):
        data = {}
        self.data.loadSelectedExperiment()
        if self.data.experiment is None or experiment_has_thresholdCache(self.data.experiment.id):
            self.recalculateDataCheckbox.setVisible(True)
        else:
            self.recalculateDataCheckbox.setVisible(False)

    def nextId(self):
        if not self.recalculateDataCheckbox.isVisible() or self.recalculateDataCheckbox.isChecked():
            return pages.THRESHOLD_CACHE
        else:
            return pages.PREVIOUS_SCORING

