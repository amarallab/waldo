__author__ = 'heltena'

from PyQt4 import QtGui


class ParamPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(ParamPage, self).__init__(parent)
        self.data = data
        self.setTitle("Parameters")


