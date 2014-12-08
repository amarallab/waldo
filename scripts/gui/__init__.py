__author__ = 'heltena'

from PyQt4 import QtGui
from page1 import WelcomePage
import pages

class WaldoApp(QtGui.QWizard):
    def __init__(self, parent=None):
        super(WaldoApp, self).__init__(parent)

        self.setPage(pages.WELCOME, WelcomePage())

        self.setField("ex_id", 0)
        self.setField("threshold", 0)
        self.setField("roi_x", 0)
        self.setField("roi_y", 0)
        self.setField("roi_r", 0)

    def closeEvent(self, ev):
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