from __future__ import absolute_import, print_function

__author__ = 'heltena'

# standard library
import os

# third party
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt
import matplotlib.pyplot as plt

# project specific

from .widgets import ExperimentResultWidget

class FinalPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(FinalPage, self).__init__(parent)

        self.data = data
        self.setTitle("Final Results")
        self.setSubTitle("")

        self.result = ExperimentResultWidget(self)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.result)
        self.setLayout(layout)

    def initializePage(self):
        self.result.initializeWidget(self.data.experiment)

    def nextId(self):
        return -1  # Final page
