__author__ = 'heltena'

from PyQt4 import QtGui
import sys
import os
import json

from gui import WaldoApp

# class Page1(QtGui.QWizardPage):
#     def __init__(self, parent=None):
#         super(Page1, self).__init__(parent)
#
#         commentLineEdit = QtGui.QLineEdit("")
#
#         layout = QtGui.QVBoxLayout()
#         layout.addWidget(commentLineEdit)
#         self.setLayout(layout)
#
#         self.setTitle("Page 1")
#         self.registerField("comment", commentLineEdit)
#
#
# class Page2(QtGui.QWizardPage):
#     def __init__(self, parent=None):
#         super(Page2, self).__init__(parent)
#         self.setTitle("Page 2")
#
#
# class Window(QtGui.QWizard):
#     def __init__(self, parent=None):
#         super(Window, self).__init__(parent)
#         self.addPage(Page1())
#         self.addPage(Page2())
#
#     def accept(self):
#         comment = self.field("comment").toString()
#         print "Hola ", comment
#         firstPage = self.page(0)
#         firstPage.show()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = WaldoApp()
    main.show()
    main.raise_()
    sys.exit(app.exec_())
