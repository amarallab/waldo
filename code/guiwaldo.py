__author__ = 'heltena'

class Blackhole(object):
    def write(self, text):
        pass
    def flush(self):
        pass

import sys
sys.stdout = Blackhole()
sys.stderr = Blackhole()
del Blackhole

import matplotlib
matplotlib.use('Qt4Agg')
from PyQt4 import QtGui
import os
import json

from waldo.gui import WaldoApp

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = WaldoApp()
    main.show()
    main.raise_()
    sys.exit(app.exec_())
