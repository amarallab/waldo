__author__ = 'heltena'

from config import ConfigurationData
from PyQt4 import QtGui

from experimentlist import ExperimentListPage
from cachethresholddata import CacheThresholdDataPage
from score import ScorePage


class WaldoApp(QtGui.QWizard):
    def __init__(self, parent=None):
        super(WaldoApp, self).__init__(parent)

        # Data
        self.data = ConfigurationData()

        self.experimentListPage = ExperimentListPage(self.data)
        self.cacheThresholdDataPage = CacheThresholdDataPage(self.data)
        self.scorePage = ScorePage(self.data)

        self.addPage(self.experimentListPage)
        self.addPage(self.cacheThresholdDataPage)
        self.addPage(self.scorePage)

        self.experimentListPage.update_experiment_list()

    def closeEvent(self, ev):
        self.data.save()

        mb = QtGui.QMessageBox()
        mb.setText("Are you sure you want to close?")
        mb.setStandardButtons(QtGui.QMessageBox.Close | QtGui.QMessageBox.Cancel)
        mb.setDefaultButton(QtGui.QMessageBox.Close);
        result = mb.exec_()
        if result == QtGui.QMessageBox.Close:
            ev.accept()
        elif result == QtGui.QMessageBox.Cancel:
            ev.ignore()
        else:
            super(WaldoApp, self).closeEvent(ev)

    def accept(self):
        self.restart()