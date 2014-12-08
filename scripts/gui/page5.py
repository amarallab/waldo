__author__ = 'heltena'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

from waldo.conf import settings


class PreviousScoringPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(PreviousScoringPage, self).__init__(parent)

        self.data = data
        self.setTitle("Scoring")
        self.setSubTitle("The next page will run the scoring process. It could be take a few minutes.")

        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)