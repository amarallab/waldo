__author__ = 'heltena'

import os

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

import threading

from waldo.conf import settings
import glob

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
                self.row_summary_changed.emit(row, has_error, summary_name, duration)

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

    def row_summary_changed(self, row, has_error, summary, duration):
        items = [None, summary, duration]
        for col, item in enumerate(items):
            cell = self.experimentTable.item(row, col)
            if cell is not None:
                if item is not None:
                    cell.setText(QtCore.QString(item))
                if has_error:
                    cell.setBackground(Qt.red)
                else:
                    cell.setBackground(Qt.white)
            else:
                print "Row is None", row

    def initializePage(self):
        self.asyncSummaryLoader.clearRows()
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
            if row in self.errorRows:
                self.data.selected_ex_id = None
            else:
                item = self.experimentTable.item(row, 0)
                if item is None:
                    self.data.selected_ex_id = None
                else:
                    self.data.selected_ex_id = str(item.text())
        self.completeChanged.emit()

    def isComplete(self):
        return self.data.selected_ex_id is not None