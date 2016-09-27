__author__ = 'heltena'

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
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from skimage import morphology
from skimage.measure import regionprops

# from waldo.images.grab_images import grab_images_in_time_range
from waldo.gui import tasking
from waldo.wio import paths
from .widgets import ThresholdCacheWidget

class ThresholdCachePage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(ThresholdCachePage, self).__init__(parent)

        self.data = data
        self.setTitle("Image Curation")

        self.thresholdCache = ThresholdCacheWidget(self.thresholdCache_changed, self)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.thresholdCache)
        self.setLayout(layout)

    def initializePage(self):
        if self.data.experiment is None:
            self.thresholdCache.clear_experiment_data()
        else:
            self.thresholdCache.load_experiment(self.data.experiment)

    def thresholdCache_changed(self):
        self.completeChanged.emit()

    def isComplete(self):
        return self.thresholdCache.isComplete()

