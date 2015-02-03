__author__ = 'heltena'

import os

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

import threading

from waldo.conf import settings
import glob

def get_summary_data(experiment_name):
    files = glob.glob(settings.MWT_DATA_ROOT + "/" + experiment_name + "/*.summary")
    summary_name = ""
    duration = ""

    if len(files) != 1:
        return False, summary_name, duration

    summary_name = os.path.splitext(os.path.basename(files[0]))[0]
    with open(files[0], "rt") as f:
        lines = f.readlines()
        if len(lines) == 0:
            return False, summary_name, duration

        lastline = lines[-1]
        params = lastline.split(' ')
        if len(params) < 2:
            return False, summary_name, duration

        duration = params[1]
    return True, summary_name, duration


class AsyncSummaryLoader(QtCore.QThread):
    row_summary_changed = QtCore.pyqtSignal([int, bool, str, str])

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.lock = threading.Lock()
        self.sleep = threading.Semaphore(1)
        self.queue = []

    def startListening(self):
        self.finish = False
        self.start()

    def run(self):
        while not self.finish:
            self.lock.acquire()
            if len(self.queue) == 0:
                self.lock.release()
                self.sleep.acquire()
            else:
                row, folder, item = self.queue.pop(0)
                self.lock.release()

                valid, summary_name, duration = get_summary_data(folder)
                self.row_summary_changed.emit(row, valid, summary_name, duration)

    def stopListening(self):
        with self.lock:
            self.queue = []
        self.sleep.release()
        self.finish = True
        self.wait()
        self.terminate()

    def clearRows(self):
        with self.lock:
            self.queue = []

    def addRow(self, row, folder, item):
        self.lock.acquire()
        if len(self.queue) == 0:
            self.sleep.release()
        self.queue.append((row, folder, item))
        self.lock.release()


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

    def gui_close_event(self):
        self.asyncSummaryLoader.stopListening()

    def row_summary_changed(self, row, valid, summary, duration):
        items = [None, summary, duration]
        for col, item in enumerate(items):
            cell = self.experimentTable.item(row, col)
            if cell is not None:
                if item is not None:
                    cell.setText(QtCore.QString(item))
                if valid:
                    cell.setBackground(Qt.white)
                else:
                    cell.setBackground(Qt.red)

    def initializePage(self):
        self.asyncSummaryLoader.clearRows()
        self.experimentTable.clear()
        self.experimentTable.setColumnCount(3)
        self.experimentTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.experimentTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.experimentTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Filename"))
        self.experimentTable.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Title"))
        self.experimentTable.setHorizontalHeaderItem(2, QtGui.QTableWidgetItem("Duration"))

        self.experimentTable.setColumnWidth(0, 150)
        self.experimentTable.setColumnWidth(1, 175)
        self.experimentTable.setColumnWidth(2, 100)

        vh = QtGui.QHeaderView(Qt.Vertical)
        vh.setResizeMode(QtGui.QHeaderView.Fixed)
        self.experimentTable.setVerticalHeader(vh)

        self.errorRows = set()
        rowToSelect = None
        folders = sorted(os.listdir(settings.MWT_DATA_ROOT))
        self.experimentTable.setRowCount(len(folders))
        for row, folder in enumerate(folders):
            item = QtGui.QTableWidgetItem(folder)
            self.asyncSummaryLoader.addRow(row, folder, item)
            summary_name = ""
            duration = ""

            items = [item, QtGui.QTableWidgetItem(summary_name), QtGui.QTableWidgetItem(duration)]
            # if has_error:
            #     self.errorRows.add(row)
            #     for item in items:
            #         item.setBackground(Qt.red)

            for col, item in enumerate(items):
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
                valid, summary, duration = get_summary_data(str(item.text()))
                self.row_summary_changed(row, valid, summary, duration)
                if valid:
                    self.data.selected_ex_id = str(item.text())
                else:
                    self.data.selected_ex_id = None
        self.completeChanged.emit()

    def isComplete(self):
        return self.data.selected_ex_id is not None