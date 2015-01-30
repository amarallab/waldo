__author__ = 'heltena'

import os

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

from waldo.conf import settings
import glob


class SelectExperimentPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(SelectExperimentPage, self).__init__(parent)

        self.data = data
        self.setTitle("Select Experiment")
        self.setSubTitle("Select the experiment you want to run. If it is not appearing, click 'back' and change "
                         "the 'Raw Data' folder.")

        self.experimentTable = QtGui.QTableWidget()
        self.experimentTable.itemSelectionChanged.connect(self.experimentTable_itemSelectionChanged)
        self.errorRows = set()

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.experimentTable)
        self.setLayout(layout)

    def initializePage(self):
        self.experimentTable.clear()
        self.experimentTable.setColumnCount(3)
        self.experimentTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.experimentTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.experimentTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Filename"))
        self.experimentTable.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Title"))
        self.experimentTable.setHorizontalHeaderItem(2, QtGui.QTableWidgetItem("Duration"))

        vh = QtGui.QHeaderView(Qt.Vertical)
        vh.setResizeMode(QtGui.QHeaderView.Fixed)
        self.experimentTable.setVerticalHeader(vh)

        self.errorRows = set()
        rowToSelect = None
        folders = sorted(os.listdir(settings.MWT_DATA_ROOT))
        self.experimentTable.setRowCount(len(folders))
        for row, folder in enumerate(folders):
            item = QtGui.QTableWidgetItem(folder)
            files = glob.glob(settings.MWT_DATA_ROOT + "/" + folder + "/*.summary")
            summary_name = ""
            duration = ""

            has_error = False
            if len(files) != 1:
                has_error = True
            else:

                summary_name = os.path.splitext(os.path.basename(files[0]))[0]
                with open(files[0], "rt") as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        has_error = True
                    else:
                        lastline = lines[-1]
                        params = lastline.split(' ')
                        if len(params) < 2:
                            has_error = True
                        else:
                            duration = params[1]

            items = [item, QtGui.QTableWidgetItem(summary_name), QtGui.QTableWidgetItem(duration)]
            if has_error:
                self.errorRows.add(row)
                for item in items:
                    item.setBackground(Qt.red)

            for col, item in enumerate(items):
                self.experimentTable.setItem(row, col, item)

            if self.data.selected_ex_id == folder:
                rowToSelect = row

        if rowToSelect is not None:
            self.experimentTable.selectRow(rowToSelect)

        self.completeChanged.emit()

        # self.experimentList.clear()
        # selectedIndex = 0
        # for dir in sorted(os.listdir(settings.MWT_DATA_ROOT)):
        #     self.experimentList.addItem(dir)
        #     if self.data.selected_ex_id == dir:
        #         selectedIndex = self.experimentList.count() - 1
        # if self.experimentList.count() > 0:
        #     self.experimentList.setCurrentRow(selectedIndex)
        # self.completeChanged.emit()

    def experimentTable_itemSelectionChanged(self):
        values = set([i.row() for i in self.experimentTable.selectedIndexes()])
        if len(values) == 1:
            row = values.pop()
            if row in self.errorRows:
                self.data.selected_ex_id = None
            else:
                item = self.experimentTable.item(row, 0)
                self.data.selected_ex_id = str(item.text())
        self.completeChanged.emit()

    def isComplete(self):
        return self.data.selected_ex_id is not None