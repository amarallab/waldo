from __future__ import absolute_import, print_function

__author__ = 'heltena'

# standard library

# third party
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

# project specific
from waldo.conf import settings
from waldo import wio
from waldo.wio import paths
from waldo.wio import file_manager as fm
from . import pages

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

        # catch prepdata.load IOerror since we only want to check
        # if report_card file exists for experiment
        try:
           report_card =  self.data.experiment.prepdata.load('report-card')
        except IOError:
           report_card = None

        if self.data.experiment is not None and report_card is not None:
            self.recalculateDataCheckbox.setVisible(True)
        else:
            self.recalculateDataCheckbox.setVisible(False)

    def nextId(self):
        if not self.recalculateDataCheckbox.isVisible() or self.recalculateDataCheckbox.isChecked():
            return pages.WALDO_PROCESS
        else:
            return pages.FINAL
