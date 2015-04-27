from __future__ import absolute_import, print_function

# standard library
import os
import glob
import json
import threading

# third party
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

# project specific
from waldo.conf import settings
from waldo.wio import paths
from . import pages
from .helpers import experiment_has_thresholdCache
from .loaders import get_summary_data, AsyncSummaryLoader


class SelectBatchModeExperimentsPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(SelectBatchModeExperimentsPage, self).__init__(parent)

        self.data = data
        self.setTitle("Select an Experiment List")
        self.setSubTitle("Select the experiment list you want to run in batch mode. If one of your experiments is not "
                         "appearing, click 'back' and change the 'Raw Data' folder.")

        self.experimentTable = QtGui.QTableWidget()
        self.experimentTable.itemSelectionChanged.connect(self.experimentTable_itemSelectionChanged)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.experimentTable)
        self.setLayout(layout)

        self.asyncSummaryLoader = AsyncSummaryLoader()
        self.asyncSummaryLoader.row_summary_changed.connect(self.row_summary_changed)
        self.asyncSummaryLoader.startListening()

        self.valid_selection = False
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
        table.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        table.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Filename"))
        table.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Title"))
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

        self.valid_selection = False
        self.errorRows = set()
        rowsToSelect = []
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

            if folder in self.data.experiment_id_list:
                rowsToSelect.append(row)

        for row in rowsToSelect:
            self.experimentTable.selectRow(row)

        self.completeChanged.emit()

    def experimentTable_itemSelectionChanged(self):
        self.data.experiment_id_list = []
        valid_selection = True
        for row in set([i.row() for i in self.experimentTable.selectedIndexes()]):
            if row in self.errorRows:
                valid_selection = False
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
                        valid_selection = False
            if valid:
                self.data.experiment_id_list.append(str(item.text()))
        self.data.no_thresholdcache_experiment_id_list = [id for id in self.data.experiment_id_list
                                                          if not experiment_has_thresholdCache(id)]
        self.valid_selection = valid_selection
        self.completeChanged.emit()

    def isComplete(self):
        return self.valid_selection and len(self.data.experiment_id_list) > 0

    def nextId(self):
        if len(self.data.no_thresholdcache_experiment_id_list) > 0:
            return pages.BATCHMODE_THRESHOLD_CACHE
        else:
            return pages.BATCHODE_WALDO_PROCESS