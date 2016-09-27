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
from waldo.wio import Experiment

from waldo import wio
import waldo.images.evaluate_acuracy as ea
#import waldo.images.worm_finder as wf
import waldo.metrics.report_card as report_card
from .widgets import ExperimentResultWidget

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



class BatchModeFinalPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(BatchModeFinalPage, self).__init__(parent)

        self.data = data
        self.setTitle("Final Results")
        self.setSubTitle("")

        self.current_experiment_id = None

        self.experimentListComboBox = QtGui.QComboBox()
        self.experimentListComboBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.experimentListComboBox.currentIndexChanged.connect(self.experimentListComboBox_currentIndexChanged)
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(QtGui.QLabel("Select experiment"))
        hbox.addWidget(self.experimentListComboBox)

        self.result = ExperimentResultWidget(self)
        layout = QtGui.QVBoxLayout()
        layout.addLayout(hbox)
        layout.addWidget(self.result)
        self.setLayout(layout)

    def initializePage(self):
        self.experimentListComboBox.clear()
        if len(self.data.experiment_id_list) > 0:
            for experiment_id in self.data.experiment_id_list:
                self.experimentListComboBox.addItem(experiment_id)
            self.result.setVisible(True)
            self.current_experiment_id = self.data.experiment_id_list[0]
            experiment = Experiment(experiment_id=self.data.experiment_id_list[0])
            self.result.initializeWidget(experiment)
        else:
            self.current_experiment_id = None
            self.result.setVisible(False)

    def experimentListComboBox_currentIndexChanged(self, index):
        experiment_id = self.data.experiment_id_list[index]
        if self.current_experiment_id != experiment_id:
            experiment = Experiment(experiment_id=experiment_id)
            self.result.setVisible(True)
            self.result.initializeWidget(experiment)
            self.current_experiment_id = experiment_id

    def nextId(self):
        return -1  # Final page
