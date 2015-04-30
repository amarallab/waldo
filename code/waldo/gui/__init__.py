from __future__ import absolute_import

__author__ = 'heltena'

# standard library
import sys, argparse

# third party
from PyQt4 import QtGui, QtCore
import numpy as np
import pandas as pd

# project specific
from waldo.wio import Experiment

from .page1 import WelcomePage
from .page2 import SelectExperimentPage
from .page3 import PreviousThresholdCachePage
from .page4 import ThresholdCachePage
from .page5 import PreviousScoringPage
from .page6 import ScoringPage
from .page7 import PreviousWaldoProcessPage
from .page8 import WaldoProcessPage
from .page9 import FinalPage

from .page10 import SelectBatchModeExperimentsPage
from .page11 import BatchModeThresholdCachePage
from .page12 import BatchModeWaldoProcessPage
from .page13 import BatchModeFinalPage

from . import pages

class WaldoAppData:
    def __init__(self):
        self.single_mode = True
        # Single mode data
        self.selected_ex_id = None
        self.experiment = None
        self.threshold = 0
        self.roi_center = (0, 0)
        self.roi_radius = 0

        # Batch mode data
        self.experiment_id_list = []
        self.no_thresholdcache_experiment_id_list = []

    def singleMode(self):
        self.single_mode = True

    def batchMode(self):
        self.single_mode = False

    def loadSelectedExperiment(self):
        if not self.single_mode:
            self.experiment = None
            return

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
        # self.data.selected_ex_id = '20130226_010927'
        # self.data.loadSelectedExperiment()

        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--batch', type=str, help="List of experiment ids")
        args = parser.parse_args()
        if args.batch is not None:
            self.data.experiment_id_list = args.batch.split(',')
        else:
            self.data.experiment_id_list = None

        self.setPage(pages.WELCOME, WelcomePage(self.data))
        self.setPage(pages.SELECT_EXPERIMENT, SelectExperimentPage(self.data))
        self.setPage(pages.PREVIOUS_THRESHOLD_CACHE, PreviousThresholdCachePage(self.data))
        self.setPage(pages.THRESHOLD_CACHE, ThresholdCachePage(self.data))
        self.setPage(pages.PREVIOUS_SCORING, PreviousScoringPage(self.data))
        self.setPage(pages.SCORING, ScoringPage(self.data))
        self.setPage(pages.PREVIOUS_WALDO_PROCESS, PreviousWaldoProcessPage(self.data))
        self.setPage(pages.WALDO_PROCESS, WaldoProcessPage(self.data))
        self.setPage(pages.FINAL, FinalPage(self.data))

        self.setPage(pages.SELECT_BATCHMODE_EXPERIMENTS, SelectBatchModeExperimentsPage(self.data))
        self.setPage(pages.BATCHMODE_THRESHOLD_CACHE, BatchModeThresholdCachePage(self.data))
        self.setPage(pages.BATCHODE_WALDO_PROCESS, BatchModeWaldoProcessPage(self.data))
        self.setPage(pages.BATCHMODE_FINAL, BatchModeFinalPage(self.data))

        self.setMinimumSize(800, 600)

    def _askForClose(self):
        mb = QtGui.QMessageBox()
        mb.setText("Are you sure you want to close?")
        mb.setStandardButtons(QtGui.QMessageBox.Close | QtGui.QMessageBox.Cancel)
        mb.setDefaultButton(QtGui.QMessageBox.Close)
        result = mb.exec_()
        if result == QtGui.QMessageBox.Close:
            return True
        elif result == QtGui.QMessageBox.Cancel:
            return False
        else:
            return None

    def closeEvent(self, ev):
        result = self._askForClose()
        if result:
            for ids in self.pageIds():
                page = self.page(ids)
                method = getattr(page, 'gui_close_event', None)
                if callable(method):
                    method()
            ev.accept()
        elif result == False:
            ev.ignore()
        else:
            super(WaldoApp, self).closeEvent(ev)

    def accept(self):
        self.restart()

    def reject(self):
        if self._askForClose():
            super(WaldoApp, self).reject()