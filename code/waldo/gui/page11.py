import os

from PyQt4 import QtGui
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt, QTimer

import numpy as np
from scipy import ndimage
import json
import errno
from waldo.wio import Experiment

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from skimage import morphology
from skimage.measure import regionprops

# from waldo.images.grab_images import grab_images_in_time_range
from waldo.gui import tasking
from waldo.wio import paths
from .widgets import ThresholdCacheWidget
from .helpers import experiment_has_thresholdCache
from . import pages


class BatchModeThresholdCachePage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(BatchModeThresholdCachePage, self).__init__(parent)

        self.data = data
        self.setTitle("Image Curation")

        self.thresholdCache = ThresholdCacheWidget(self.thresholdCache_changed, self)
        self.experimentLabel = QtGui.QLabel("")
        self.nextButton = QtGui.QPushButton("Next")
        self.nextButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.nextButton.clicked.connect(self.nextButton_clicked)

        buttons = QtGui.QHBoxLayout()
        buttons.addWidget(self.experimentLabel)
        buttons.addWidget(self.nextButton)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.thresholdCache)
        layout.addLayout(buttons)
        self.setLayout(layout)
        self.current_index = 0
        self.valids = {}

    def initializePage(self):
        if self.data.experiment_id_list is None:
            self.data.experiment_id_list = []

        if len(self.data.no_thresholdcache_experiment_id_list) > 0:
            self.current_index = 0
            self.load_current_experiment()
        else:
            self.thresholdCache.clear_experiment_data()

    def load_current_experiment(self):
        experiment_id = self.data.no_thresholdcache_experiment_id_list[self.current_index]
        if len(self.data.no_thresholdcache_experiment_id_list) > 1:
            suffix = "(remain: {})".format(len(self.data.no_thresholdcache_experiment_id_list) - 1)
        else:
            suffix = ""
        self.experimentLabel.setText("Experiment: {} {}".format(experiment_id, suffix))
        self.experiment = Experiment(experiment_id=experiment_id)
        self.thresholdCache.load_experiment(self.experiment)
        self.nextButton.setEnabled(False)

    def thresholdCache_changed(self):
        if experiment_has_thresholdCache(self.experiment.id):
            try:
                self.data.no_thresholdcache_experiment_id_list.remove(self.experiment.id)
            except:
                print("Warning: %d not found" % self.experiment.id)
            self.nextButton.setEnabled(len(self.data.no_thresholdcache_experiment_id_list) > 0)
        self.completeChanged.emit()

    def nextButton_clicked(self):
        if len(self.data.no_thresholdcache_experiment_id_list) > 0:
            self.load_current_experiment()

    def isComplete(self):
        return len(self.data.no_thresholdcache_experiment_id_list) == 0
