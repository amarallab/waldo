__author__ = 'heltena'

import os

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

from waldo.conf import settings


class SelectExperimentPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(SelectExperimentPage, self).__init__(parent)

        self.data = data
        self.setTitle("Select Experiment")
        self.setSubTitle("Select the experiment you want to run. If it is not appearing, click 'back' and change "
                         "the 'Raw Data' folder.")

        self.experimentList = QtGui.QListWidget()
        self.experimentList.currentRowChanged.connect(self.experimentList_currentRowChanged)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.experimentList)
        self.setLayout(layout)

    def initializePage(self):
        self.experimentList.clear()
        selectedIndex = 0
        for name, dirs, files in os.walk(settings.MWT_DATA_ROOT):
            for dir in dirs:
                self.experimentList.addItem(dir)
                if self.data.selected_ex_id == dir:
                    selectedIndex = self.experimentList.count() - 1
        if self.experimentList.count() > 0:
            self.experimentList.setCurrentRow(selectedIndex)
        self.completeChanged.emit()

    def experimentList_currentRowChanged(self):
        if self.experimentList.currentItem() is not None:
            self.data.selected_ex_id = str(self.experimentList.currentItem().text())
        self.completeChanged.emit()

    def isComplete(self):
        return self.data.selected_ex_id is not None