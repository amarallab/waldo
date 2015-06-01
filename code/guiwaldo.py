__author__ = 'heltena'


class LogForWindows(object):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'at')

    def write(self, text):
        self.file.write(text)

    def flush(self):
        self.file.flush()

import sys
log = LogForWindows('guiwaldo_log.txt')
sys.stdout = log
sys.stderr = log
del LogForWindows

import matplotlib
matplotlib.use('Qt4Agg')
from PyQt4 import QtGui
# import os
# import json
from waldo.gui import WaldoApp


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = WaldoApp()
    main.show()
    main.raise_()
    sys.exit(app.exec_())
