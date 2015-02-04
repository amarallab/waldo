__author__ = 'heltena'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt
import pages

from waldo.conf import settings

from waldo import wio
from waldo.wio import paths
from waldo.wio import file_manager as fm


class PreviousWaldoProcessPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(PreviousWaldoProcessPage, self).__init__(parent)

        self.data = data
        self.setTitle("Waldo Process")
        self.setSubTitle("The next page will start running the waldo process. It could take a few minutes.")

        self.recalculateDataCheckbox = QtGui.QCheckBox("Recalculate data.")
        self.recalculateDataCheckbox.setVisible(False)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.recalculateDataCheckbox)
        self.setLayout(layout)

    def initializePage(self):
        self.data.loadSelectedExperiment()
        if self.data.experiment is not None and self.data.experiment.prepdata.load('report-card') is not None:
            self.recalculateDataCheckbox.setVisible(True)
        else:
            self.recalculateDataCheckbox.setVisible(False)

    def nextId(self):
        if not self.recalculateDataCheckbox.isVisible() or self.recalculateDataCheckbox.isChecked():
            return pages.WALDO_PROCESS
        else:
            return pages.FINAL
