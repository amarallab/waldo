__author__ = 'heltena'

from PyQt4 import QtGui
import sys
import os
import json

from gui import WaldoApp

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = WaldoApp()
    main.show()
    main.raise_()
    sys.exit(app.exec_())
