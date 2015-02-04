__author__ = 'heltena'

from waldo.wio import Experiment

from PyQt4 import QtGui
from page1 import WelcomePage
from page2 import SelectExperimentPage
from page3 import PreviousThresholdCachePage
from page4 import ThresholdCachePage
from page5 import PreviousScoringPage
from page6 import ScoringPage
from page7 import PreviousWaldoProcessPage
from page8 import WaldoProcessPage
from page9 import FinalPage

import pages

class WaldoAppData:
    def __init__(self):
        self.selected_ex_id = None
        self.experiment = None
        self.threshold = 0
        self.roi_center = (0, 0)
        self.roi_radius = 0

    def loadSelectedExperiment(self):
        if self.selected_ex_id is not None:
            self.loadExperiment(self.selected_ex_id)
        else:
            self.experiment = None

    def loadExperiment(self, ex_id):
        self.experiment = Experiment(experiment_id=ex_id)

class WaldoApp(QtGui.QWizard):
    def __init__(self, parent=None):
        super(WaldoApp, self).__init__(parent)

        self.data = WaldoAppData()
        # self.data.selected_ex_id = '20141017_123722'
        # self.data.loadSelectedExperiment()
        #
        self.setPage(pages.WELCOME, WelcomePage(self.data))
        self.setPage(pages.SELECT_EXPERIMENT, SelectExperimentPage(self.data))
        self.setPage(pages.PREVIOUS_THRESHOLD_CACHE, PreviousThresholdCachePage(self.data))
        self.setPage(pages.THRESHOLD_CACHE, ThresholdCachePage(self.data))
        self.setPage(pages.PREVIOUS_SCORING, PreviousScoringPage(self.data))
        self.setPage(pages.SCORING, ScoringPage(self.data))
        self.setPage(pages.PREVIOUS_WALDO_PROCESS, PreviousWaldoProcessPage(self.data))
        self.setPage(pages.WALDO_PROCESS, WaldoProcessPage(self.data))
        self.setPage(pages.FINAL, FinalPage(self.data))

    def closeEvent(self, ev):
        mb = QtGui.QMessageBox()
        mb.setText("Are you sure you want to close?")
        mb.setStandardButtons(QtGui.QMessageBox.Close | QtGui.QMessageBox.Cancel)
        mb.setDefaultButton(QtGui.QMessageBox.Close);
        result = mb.exec_()
        if result == QtGui.QMessageBox.Close:
            for ids in self.pageIds():
                page = self.page(ids)
                method = getattr(page, 'gui_close_event', None)
                if callable(method):
                    method()
            ev.accept()
        elif result == QtGui.QMessageBox.Cancel:
            ev.ignore()
        else:
            super(WaldoApp, self).closeEvent(ev)

    def accept(self):
        self.restart()