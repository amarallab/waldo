__author__ = 'heltena'

import os
import json
import pages

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

from waldo.wio import paths


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
        if self.data.experiment is not None:
            self.annotation_filename = paths.threshold_data(self.data.experiment.id)
            try:
                with open(str(self.annotation_filename), "rt") as f:
                    data = json.loads(f.read())
            except IOError as ex:
                pass

        if 'threshold' in data and 'r' in data and 'x' in data and 'y' in data:
            self.recalculateDataCheckbox.setVisible(True)
        else:
            self.recalculateDataCheckbox.setVisible(False)

    def nextId(self):
        if not self.recalculateDataCheckbox.isVisible() or self.recalculateDataCheckbox.isChecked():
            return pages.THRESHOLD_CACHE
        else:
            return pages.PREVIOUS_SCORING
