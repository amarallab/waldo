__author__ = 'heltena'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt
import pages

from waldo.conf import settings


class PreviousScoringPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(PreviousScoringPage, self).__init__(parent)

        self.data = data
        self.setTitle("Scoring")
        self.setSubTitle("The next page will run the scoring process. It could take a few minutes.")

        self.recalculateDataCheckbox = QtGui.QCheckBox("Recalculate data.")
        self.recalculateDataCheckbox.setVisible(True)
        self.recalculateDataCheckbox.setChecked(True)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.recalculateDataCheckbox)
        self.setLayout(layout)

    def nextId(self):
        if not self.recalculateDataCheckbox.isVisible() or self.recalculateDataCheckbox.isChecked():
            return pages.SCORING
        else:
            return pages.PREVIOUS_WALDO_PROCESS
