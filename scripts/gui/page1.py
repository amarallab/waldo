__author__ = 'heltena'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

from waldo.conf import settings


class WelcomePage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(WelcomePage, self).__init__(parent)

        self.data = data
        self.setTitle("Welcome")
        self.setSubTitle("Welcome to Waldo GUI. Press 'configure' to configure parameters, 'next' to continue")

        config_button = QtGui.QPushButton("Configure")
        config_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        config_button.clicked.connect(self.config_button_clicked)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(config_button)
        self.setLayout(layout)

    def config_button_clicked(self):
        dlg = ConfigDialog(self)
        dlg.setModal(True)
        dlg.exec_()

class ConfigDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        super(ConfigDialog, self).__init__(parent)

        settings.load()
        layout = QtGui.QGridLayout()

        # Collider Suite
        self.colliderSuiteAssimilateSize = self.createQLineEditIntValidator(str(settings.COLLIDER_SUITE_ASSIMILATE_SIZE),
                                                                            settings.COLLIDER_SUITE_ASSIMILATE_SIZE_RANGE)
        self.colliderSuiteOffshoot = self.createQLineEditIntValidator(str(settings.COLLIDER_SUITE_OFFSHOOT),
                                                                      settings.COLLIDER_SUITE_OFFSHOOT_RANGE)
        self.colliderSuiteSplitAbs = self.createQLineEditIntValidator(str(settings.COLLIDER_SUITE_SPLIT_ABS),
                                                                      settings.COLLIDER_SUITE_SPLIT_ABS_RANGE)
        self.colliderSuiteSplitRel = self.createQLineEditDoubleValidator(str(settings.COLLIDER_SUITE_SPLIT_REL),
                                                                         settings.COLLIDER_SUITE_SPLIT_REL_RANGE)

        col = 0
        layout.addWidget(QtGui.QLabel("<b>Collider suite</b>"), col, 0, 1, 2)

        col += 1
        layout.addWidget(QtGui.QLabel("Assimilate size"), col, 0, 1, 1)
        layout.addWidget(self.colliderSuiteAssimilateSize, col, 1, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("Offshoot"), col, 0, 1, 1)
        layout.addWidget(self.colliderSuiteOffshoot, col, 1, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("Split Abs"), col, 0, 1, 1)
        layout.addWidget(self.colliderSuiteSplitAbs, col, 1, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("Split Rel"), col, 0, 1, 1)
        layout.addWidget(self.colliderSuiteSplitRel, col, 1, 1, 1)

        # Tape
        self.tapeFrameSearchLimit = self.createQLineEditIntValidator(str(settings.TAPE_FRAME_SEARCH_LIMIT),
                                                                     settings.TAPE_FRAME_SEARCH_LIMIT_RANGE)
        self.tapeKdeSamples = self.createQLineEditIntValidator(str(settings.TAPE_KDE_SAMPLES),
                                                               settings.TAPE_KDE_SAMPLES_RANGE)
        self.tapeMaxSpeedMultiplier = self.createQLineEditDoubleValidator(str(settings.TAPE_MAX_SPEED_MULTIPLIER),
                                                                          settings.TAPE_MAX_SPEED_MULTIPLIER_RANGE)
        self.tapeMaxSpeedSmoothing = self.createQLineEditIntValidator(str(settings.TAPE_MAX_SPEED_SMOOTHING),
                                                                      settings.TAPE_MAX_SPEED_SMOOTHING_RANGE)
        self.tapeMinTraceFail = self.createQLineEditIntValidator(str(settings.TAPE_MIN_TRACE_FAIL),
                                                                 settings.TAPE_MIN_TRACE_FAIL_RANGE)
        self.tapeMinTraceWarn = self.createQLineEditIntValidator(str(settings.TAPE_MIN_TRACE_WARN),
                                                                 settings.TAPE_MIN_TRACE_WARN_RANGE)
        self.tapeRelMoveThreshold = self.createQLineEditDoubleValidator(str(settings.TAPE_REL_MOVE_THRESHOLD),
                                                                        settings.TAPE_REL_MOVE_THRESHOLD_RANGE)
        self.tapeShakycamAllowance = self.createQLineEditIntValidator(str(settings.TAPE_SHAKYCAM_ALLOWANCE),
                                                                      settings.TAPE_SHAKYCAM_ALLOWANCE_RANGE)
        self.tapeTraceLimitNum = self.createQLineEditIntValidator(str(settings.TAPE_TRACE_LIMIT_NUM),
                                                                  settings.TAPE_TRACE_LIMIT_NUM_RANGE)

        col = 0
        layout.addWidget(QtGui.QLabel("<b>Tape</b>"), col, 2, 1, 2)

        col += 1
        layout.addWidget(QtGui.QLabel("Frame Search Limit"), col, 2, 1, 1)
        layout.addWidget(self.tapeFrameSearchLimit, col, 3, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("KDE Samples"), col, 2, 1, 1)
        layout.addWidget(self.tapeKdeSamples, col, 3, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("Max Speed Multiplier"), col, 2, 1, 1)
        layout.addWidget(self.tapeMaxSpeedMultiplier, col, 3, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("Max Speed Smoothing"), col, 2, 1, 1)
        layout.addWidget(self.tapeMaxSpeedSmoothing, col, 3, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("Min Trace Fail"), col, 2, 1, 1)
        layout.addWidget(self.tapeMinTraceFail, col, 3, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("Min Trace Warn"), col, 2, 1, 1)
        layout.addWidget(self.tapeMinTraceWarn, col, 3, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("Rel Move Threshold"), col, 2, 1, 1)
        layout.addWidget(self.tapeRelMoveThreshold, col, 3, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("Shakycam Allowance"), col, 2, 1, 1)
        layout.addWidget(self.tapeShakycamAllowance, col, 3, 1, 1)

        col += 1
        layout.addWidget(QtGui.QLabel("Trace Limit Num"), col, 2, 1, 1)
        layout.addWidget(self.tapeTraceLimitNum, col, 3, 1, 1)

        # Folders
        folderLayout = QtGui.QGridLayout()

        self.rawDataLabel = QtGui.QLabel(settings.MWT_DATA_ROOT)
        rawDataButton = QtGui.QPushButton("Change")
        rawDataButton.clicked.connect(self.rawDataButton_clicked)

        self.waldoDataLabel = QtGui.QLabel(settings.PROJECT_DATA_ROOT)
        waldoDataButton = QtGui.QPushButton("Change")
        waldoDataButton.clicked.connect(self.waldoDataButton_clicked)

        col = 0
        folderLayout.addWidget(QtGui.QLabel("<b>Folders</b>"), col, 0, 1, 3)

        col += 1
        folderLayout.addWidget(QtGui.QLabel("Raw Data"), col, 0, 1, 1)
        folderLayout.addWidget(self.rawDataLabel, col, 1, 1, 1)
        folderLayout.addWidget(rawDataButton, col, 2, 1, 1)

        col += 1
        folderLayout.addWidget(QtGui.QLabel("Project Data"), col, 0, 1, 1)
        folderLayout.addWidget(self.waldoDataLabel, col, 1, 1, 1)
        folderLayout.addWidget(waldoDataButton, col, 2, 1, 1)

        # Buttons
        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Save|QtGui.QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.button(QtGui.QDialogButtonBox.Save).clicked.connect(self.save_clicked)
        buttons.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.cancel_clicked)
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addLayout(layout)
        mainLayout.addLayout(folderLayout)
        mainLayout.addWidget(buttons)
        self.setLayout(mainLayout)
        self.setFixedSize(self.minimumSize())
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.CustomizeWindowHint)

    def createQLineEditIntValidator(self, value, range):
        result = QtGui.QLineEdit(value)
        result.setValidator(QtGui.QIntValidator(range[0], range[1], self))
        return result

    def createQLineEditDoubleValidator(self, value, range):
        result = QtGui.QLineEdit(value)
        result.setValidator(QtGui.QDoubleValidator(range[0], range[1], range[2], self))
        return result

    def rawDataButton_clicked(self, ev):
        result = str(QtGui.QFileDialog.getExistingDirectory(directory=self.rawDataLabel.text()))
        if len(result) > 0:
            self.rawDataLabel.setText(result)

    def waldoDataButton_clicked(self, ev):
        result = str(QtGui.QFileDialog.getExistingDirectory(directory=self.waldoDataLabel.text()))
        if len(result) > 0:
            self.waldoDataLabel.setText(result)

    def _intValueOf(self, lineEdit, currentValue):
        value, result = lineEdit.text().toInt()
        return value if result else currentValue

    def _doubleValueOf(self, lineEdit, currentValue):
        value, result = lineEdit.text().toDouble()
        return value if result else currentValue

    def save_clicked(self, ev):
        settings.COLLIDER_SUITE_ASSIMILATE_SIZE = self._intValueOf(self.colliderSuiteAssimilateSize, settings.COLLIDER_SUITE_ASSIMILATE_SIZE)
        settings.COLLIDER_SUITE_OFFSHOOT = self._intValueOf(self.colliderSuiteOffshoot, settings.COLLIDER_SUITE_OFFSHOOT)
        settings.COLLIDER_SUITE_SPLIT_ABS = self._intValueOf(self.colliderSuiteSplitAbs, settings.COLLIDER_SUITE_SPLIT_ABS)
        settings.COLLIDER_SUITE_SPLIT_REL = self._doubleValueOf(self.colliderSuiteSplitRel, settings.COLLIDER_SUITE_SPLIT_REL)

        settings.TAPE_REL_MOVE_THRESHOLD = self._doubleValueOf(self.tapeRelMoveThreshold, settings.TAPE_REL_MOVE_THRESHOLD)
        settings.TAPE_MIN_TRACE_FAIL = self._intValueOf(self.tapeMinTraceFail, settings.TAPE_MIN_TRACE_FAIL)
        settings.TAPE_MIN_TRACE_WARN = self._intValueOf(self.tapeMinTraceWarn, settings.TAPE_MIN_TRACE_WARN)
        settings.TAPE_TRACE_LIMIT_NUM = self._intValueOf(self.tapeTraceLimitNum, settings.TAPE_TRACE_LIMIT_NUM)
        settings.TAPE_FRAME_SEARCH_LIMIT = self._intValueOf(self.tapeFrameSearchLimit, settings.TAPE_FRAME_SEARCH_LIMIT)
        settings.TAPE_KDE_SAMPLES = self._intValueOf(self.tapeKdeSamples, settings.TAPE_KDE_SAMPLES)
        settings.TAPE_MAX_SPEED_MULTIPLIER = self._doubleValueOf(self.tapeMaxSpeedMultiplier, settings.TAPE_MAX_SPEED_MULTIPLIER)
        settings.TAPE_SHAKYCAM_ALLOWANCE = self._intValueOf(self.tapeShakycamAllowance, settings.TAPE_SHAKYCAM_ALLOWANCE)
        settings.TAPE_MAX_SPEED_SMOOTHING = self._intValueOf(self.tapeMaxSpeedSmoothing, settings.TAPE_MAX_SPEED_SMOOTHING)

        settings.MWT_DATA_ROOT = str(self.rawDataLabel.text())
        settings.PROJECT_DATA_ROOT = str(self.waldoDataLabel.text())

        settings.save()
        self.close()

    def cancel_clicked(self, ev):
        self.close()
