__author__ = 'heltena'

import os

from PyQt4 import QtGui
from PyQt4.QtGui import QSizePolicy

class ExperimentListPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(ExperimentListPage, self).__init__(parent)
        self.data = data
        self.setTitle("Experiments Folder")
        self.setSubTitle("Select the experiment you want to use")

        # First Row
        experiment_folder_label = QtGui.QLabel("Experiments Folder")
        experiment_folder_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        change_folder_button = QtGui.QPushButton("Change")
        change_folder_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        change_folder_button.clicked.connect(self.change_folder_clicked)

        first_row_layout = QtGui.QHBoxLayout()
        first_row_layout.addWidget(experiment_folder_label)
        first_row_layout.addWidget(change_folder_button)

        # Middle row
        experiment_list_widget = QtGui.QListWidget()

        layout = QtGui.QVBoxLayout()
        layout.addLayout(first_row_layout)
        layout.addWidget(experiment_list_widget)
        self.setLayout(layout)

        self.experiment_folder_label = experiment_folder_label
        self.experiment_list = experiment_list_widget
        self.experiment_list.currentRowChanged.connect(self.completeChanged)

        self.update_experiment_list()

    def initializePage(self):
        self.experimentListPage.update_experiment_list()

    def change_folder_clicked(self):
        result = str(QtGui.QFileDialog.getExistingDirectory(directory=self.data.waldo_folder))
        if len(result) > 0:
            valid = False
            for name, dirs, files in os.walk(result):
                if 'raw_data' in dirs:
                    valid = True
                break

            if not valid:
                mb = QtGui.QMessageBox()
                mb.setText("This folder doesn't contain a 'raw_data' folder.")
                mb.exec_()
            else:
                self.data.waldo_folder = result
                self.data.save()
                self.update_experiment_list()

    def update_experiment_list(self):
        self.experiment_folder_label.setText(self.data.waldo_folder)
        self.experiment_list.clear()
        for name, dirs, files in os.walk(os.path.join(self.data.waldo_folder, 'raw_data')):
            for dir in dirs:
                self.experiment_list.addItem(dir)
            break

    def isComplete(self):
        return 0 <= self.experiment_list.currentRow() < self.experiment_list.count()