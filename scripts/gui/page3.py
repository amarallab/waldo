__author__ = 'heltena'

import os
import json
import pages

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

from waldo.conf import settings


class PreviousThresholdCachePage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(PreviousThresholdCachePage, self).__init__(parent)

        self.data = data
        self.setTitle("Threshold Cache")
        self.setSubTitle("The next page will load the threshold cache data. It could be take a few minutes.")

        self.skipCheckbox = QtGui.QCheckBox("Skip step")
        self.skipCheckbox.setVisible(False)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.skipCheckbox)
        self.setLayout(layout)

    def initializePage(self):
        data = {}
        if self.data.ex_id is not None:
            self.annotation_filename = os.path.join(settings.PROJECT_DATA_ROOT, self.data.ex_id, 'thresholddata.json')
            try:
                with open(self.annotation_filename, "rt") as f:
                    data = json.loads(f.read())
            except IOError as ex:
                pass

        if 'threshold' in data and 'r' in data and 'x' in data and 'y' in data:
            self.skipCheckbox.setVisible(True)
        else:
            self.skipCheckbox.setVisible(False)

    def nextId(self):
        if self.skipCheckbox.isChecked():
            return pages.PREVIOUS_SCORING
        else:
            return pages.THRESHOLD_CACHE
