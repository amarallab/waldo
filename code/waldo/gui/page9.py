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

from waldo import wio
import waldo.images.evaluate_acuracy as ea
#import waldo.images.worm_finder as wf
import waldo.metrics.report_card as report_card
from .appdata import WaldoAppData, WaldoBatchRunResult

#import waldo.metrics.step_simulation as ssim
#import waldo.viz.eye_plots as ep

#from waldo.gui import pathcustomize
#import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
#from mpltools import style

#import matplotlib.image as mpimg
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
#from skimage import morphology
#from skimage.measure import regionprops

#style.use('ggplot')


from .widgets import ExperimentResultWidget

class FinalPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(FinalPage, self).__init__(parent)

        self.data = data
        self.setTitle("Final Results")
        self.setSubTitle("")

        self.message = QtGui.QLabel("Select an experiment")
        self.result = ExperimentResultWidget(self)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.message)
        layout.addWidget(self.result)
        self.setLayout(layout)

    def initializePage(self):
        if self.data.single_result_message is None:
            state = WaldoBatchRunResult.CACHED
            param = None
        else:
            state, param = self.data.single_result_message

        showExperiment = True
        if state == WaldoBatchRunResult.CACHED:
            self.message.setVisible(True)
            self.message.setText("Using cache data.")
            self.message.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        elif state == WaldoBatchRunResult.SUCCEEDED:
            self.message.setVisible(False)
        else:
            tb, ex = param
            self.message.setVisible(True)
            self.message.setText("<font color=\"red\">Failed:</font> {}".format(ex))
            self.message.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            showExperiment = False
            print("Trace: {}".format(ex))
            print("End Trace.")

        if showExperiment:
            self.result.setVisible(True)
            self.result.initializeWidget(self.data.experiment)
        else:
            self.result.setVisible(False)

    def nextId(self):
        return -1  # Final page
