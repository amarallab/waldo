from __future__ import absolute_import, print_function

__author__ = 'heltena'

# standard library
import os
import json

# third party
from PyQt4 import QtGui, QtCore
# from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

# project specific
from waldo.conf import settings
from waldo.wio import paths
from . import pages
from .loaders import get_summary_data, AsyncSummaryLoader


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

        self.asyncSummaryLoader = AsyncSummaryLoader()
        self.asyncSummaryLoader.row_summary_changed.connect(self.row_summary_changed)
        self.asyncSummaryLoader.startListening()

        self.loadedRows = set()
        self.errorRows = set()

    def gui_close_event(self):
        self.asyncSummaryLoader.stopListening()

    def row_summary_changed(self, row, state, summary, duration):
        if row in self.loadedRows:
            return
        self.loadedRows.add(row)

        color = {AsyncSummaryLoader.ROW_STATE_INVALID: Qt.red,
                 AsyncSummaryLoader.ROW_STATE_NOT_LOADED: Qt.white,
                 AsyncSummaryLoader.ROW_STATE_VALID: Qt.white,
                 AsyncSummaryLoader.ROW_STATE_VALID_HAS_RESULTS: Qt.green}[state]

        if state == AsyncSummaryLoader.ROW_STATE_INVALID:
            self.errorRows.add(row)
        items = [None, summary, duration]
        for col, item in enumerate(items):
            cell = self.experimentTable.item(row, col)
            if cell is not None:
                if item is not None:
                    cell.setText(QtCore.QString(item))
                cell.setBackground(color)

    def _setup_table_headers(self, table):
        table.setColumnCount(3)
        table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        table.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Directory"))
        table.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Name"))
        table.setHorizontalHeaderItem(2, QtGui.QTableWidgetItem("Duration"))

        table.setColumnWidth(0, 150)
        table.setColumnWidth(1, 175)
        table.setColumnWidth(2, 100)

        vh = QtGui.QHeaderView(Qt.Vertical)
        vh.setResizeMode(QtGui.QHeaderView.Fixed)
        table.setVerticalHeader(vh)

    def initializePage(self):
        self.asyncSummaryLoader.clearRows()
        self.loadedRows = set()

        self.experimentTable.clear()
        self._setup_table_headers(self.experimentTable)

        self.errorRows = set()
        rowToSelect = None
        folders = sorted(os.listdir(settings.MWT_DATA_ROOT), reverse=True)
        self.experimentTable.setRowCount(len(folders))
        for row, folder in enumerate(folders):
            item = QtGui.QTableWidgetItem(folder)
            self.asyncSummaryLoader.addRow(row, folder, item)
            summary_name = ""
            duration = ""

            items = [item, QtGui.QTableWidgetItem(summary_name), QtGui.QTableWidgetItem(duration)]
            for col, item in enumerate(items):
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.experimentTable.setItem(row, col, item)

            if self.data.selected_ex_id == folder:
                rowToSelect = row

        if rowToSelect is not None:
            self.experimentTable.selectRow(rowToSelect)

        self.completeChanged.emit()

    def experimentTable_itemSelectionChanged(self):
        values = set([i.row() for i in self.experimentTable.selectedIndexes()])
        if len(values) == 1:
            row = values.pop()
            item = self.experimentTable.item(row, 0)
            valid = False

            if item is None:
                valid = False
                self.data.selected_ex_id = None
            else:
                if row in self.loadedRows:
                    valid = row not in self.errorRows
                else:
                    valid, summary, duration = get_summary_data(str(item.text()))
                    self.row_summary_changed(row, valid, summary, duration)
                    self.loadedRows.add(row)
                    if not valid:
                        self.errorRows.add(row)
                if valid:
                    self.data.selected_ex_id = str(item.text())
                else:
                    self.data.selected_ex_id = None
        self.completeChanged.emit()

    def isComplete(self):
        return self.data.selected_ex_id is not None

    def nextId(self):
        data = {}
        self.data.loadSelectedExperiment()
        if self.data.experiment is not None:
            self.annotation_filename = paths.threshold_data(self.data.experiment.id)
            try:
                with open(str(self.annotation_filename), "rt") as f:
                    data = json.loads(f.read())
            except IOError as ex:
                pass

        type = data.get('type', 'circle')
        if 'threshold' in data and \
                (  (type == 'circle' and 'r' in data and 'x' in data and 'y' in data) \
                or (type == 'polygon' and 'roi_points' in data)):
            return pages.PREVIOUS_THRESHOLD_CACHE
        else:
            return pages.THRESHOLD_CACHE
