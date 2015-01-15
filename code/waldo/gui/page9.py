__author__ = 'heltena'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

from waldo.conf import settings


class FinalPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(FinalPage, self).__init__(parent)

        self.data = data
        self.setTitle("Final Page")
        self.setSubTitle("TO-DO.")

        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)