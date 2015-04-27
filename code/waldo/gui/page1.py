from __future__ import absolute_import, print_function

__author__ = 'heltena'

# standard library

# third party
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

# project specific
from waldo.conf import settings, guisettings
from . import pages



#HELIO
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpltools import style

import matplotlib.image as mpimg
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from skimage import morphology
from skimage.measure import regionprops

style.use('ggplot')
#ENDHELIO

class WelcomePage(QtGui.QWizardPage):
    class Tooltips:
        config_button = "Configure"

    def __init__(self, data, parent=None):
        super(WelcomePage, self).__init__(parent)

        self.data = data
        self.setTitle("Welcome")
        self.setSubTitle("Welcome to Waldo GUI. Press 'configure' to configure parameters, 'next' to continue")

        config_button = QtGui.QPushButton("Configure")
        config_button.setToolTip(self.Tooltips.config_button)
        config_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        config_button.clicked.connect(self.config_button_clicked)

        folderLayout = QtGui.QGridLayout()

        self.rawDataLabel = QtGui.QLabel(settings.MWT_DATA_ROOT)
        rawDataButton = QtGui.QPushButton("Change")
        rawDataButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        rawDataButton.clicked.connect(self.rawDataButton_clicked)

        self.waldoDataLabel = QtGui.QLabel(settings.PROJECT_DATA_ROOT)
        waldoDataButton = QtGui.QPushButton("Change")
        waldoDataButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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

        self.runBatchModeCheckBox = QtGui.QCheckBox("Run in Batch Mode")
        self.runBatchModeCheckBox.setChecked(self.data.experiment_id_list is not None)

        #BEGIN 2015-apr-22 TEST

        self.report_figure = plt.figure()
        self.report_canvas = FigureCanvas(self.report_figure)
        self.report_canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.report_canvas.setMinimumSize(50, 50)
        self.ax_report = self.report_figure.add_subplot(111)

        df = pd.DataFrame([['P1',   4,   3,   2,    4,   7,   2,    7,   4,   6,  217,   3,  10],
                           ['P2',   5,   7,   2,  270, 310,   5,    5,   8, 301,    5,   4,   3],
                           ['P3', 389, 370, 261,    4,   7, 355,    5, 424,   5,    3, 414,  10],
                           ['P4',   4,   4,   3,    5,   4,   5,  349,   4,   3,    3,   6, 368],
                           ['P5',   4,   4,   5,    5,   6,   6,    8,   3,   4,    6,   2,   5],
                           ['P6',   2,   3,   2,    3,   3,   1,    4,   1,   2,    8,   1,   3],
                           ['P7',   5,   5,   4,    6,   2,   7,  436,   7,   5,    7,   6, 406]],
                          columns=['ID',
                                   'P1_A', 'P1_B', 'P1_C', 'P2_A', 'P2_B', 'P2_C',
                                   'P3_A', 'P3_B', 'P3_C', 'P4_A', 'P4_B', 'P4_C'])

        self.ax_report.clear()
        df.plot(ax=self.ax_report, kind='barh')
        self.ax_report.set_xlabel('Number of Track Fragments', size=15)
        self.ax_report.legend(frameon=False, loc=(1.01, 0.7), fontsize=15)
        self.ax_report.set_ylabel('phase', size=15)
        self.report_figure.tight_layout()
        self.report_figure.subplots_adjust(right=0.75)

        self.tmp_button = QtGui.QPushButton('Change!')
        self.tmp_button.clicked.connect(self.tmp_button_clicked)

        #END 2015-apr-22 TEST

        layout = QtGui.QVBoxLayout()
        layout.addWidget(config_button)
        layout.addLayout(folderLayout)
        layout.addWidget(self.runBatchModeCheckBox)
        layout.addWidget(self.report_canvas)
        layout.addWidget(self.tmp_button)
        self.setLayout(layout)

    def tmp_button_clicked(self):
        df = pd.DataFrame([['P1',   4,   3,   2,    4,   7,   2,    7,   4,   6,  217,   3,  10],
                           ['P2',   5,   7,   2,  270, 310,   5,    5,   8, 301,    5,   4,   3],
                           ['P3', 389, 370, 261,    4,   7, 355,    5, 424,   5,    3, 414,  10],
                           ['P7',   5,   5,   4,    6,   2,   7,  436,   7,   5,    7,   6, 406]],
                          columns=['ID',
                                   'P1_A', 'P1_B', 'P1_C', 'P2_A', 'P2_B', 'P2_C',
                                   'P3_A', 'P3_B', 'P3_C', 'P4_A', 'P4_B', 'P4_C'])


        self.ax_report.clear()
        df.plot(ax=self.ax_report, kind='barh')
        self.ax_report.set_xlabel('Number of Track Fragments', size=15)
        self.ax_report.legend(frameon=False, loc=(1.01, 0.7), fontsize=15)
        self.ax_report.set_ylabel('phase', size=15)
        self.report_figure.tight_layout()
        self.report_figure.subplots_adjust(right=0.75)
        self.report_figure.canvas.draw()

    def config_button_clicked(self):
        dlg = ConfigDialog(self)
        dlg.setModal(True)
        dlg.exec_()

    def rawDataButton_clicked(self, ev):
        result = str(QtGui.QFileDialog.getExistingDirectory(directory=self.rawDataLabel.text()))
        if len(result) > 0:
            self.rawDataLabel.setText(result)
            settings.MWT_DATA_ROOT = str(self.rawDataLabel.text())
            settings.save()

    def waldoDataButton_clicked(self, ev):
        result = str(QtGui.QFileDialog.getExistingDirectory(directory=self.waldoDataLabel.text()))
        if len(result) > 0:
            self.waldoDataLabel.setText(result)
            settings.PROJECT_DATA_ROOT = str(self.waldoDataLabel.text())
            settings.save()

    def nextId(self):
        # return pages.FINAL
        if self.runBatchModeCheckBox.isChecked():
            self.data.batchMode()
            return pages.SELECT_BATCHMODE_EXPERIMENTS
        else:
            self.data.singleMode()
            return pages.SELECT_EXPERIMENT


class ConfigDialog(QtGui.QDialog):
    class ToolTips:
        colliderSuiteAssimilateSize = "Collider Suite Assimilate Size"
        colliderSuiteOffshoot = "Collider suite offshoot"
        colliderSuiteSplitAbs = "Collider Suite Split Abs"
        colliderSuiteSplitRel = "Collider Suite Split Rel"

        tapeFrameSearchLimit = "Tape Frame Search Limit"
        tapeKdeSamples = "Tape Kde Samples"
        tapeMaxSpeedMultiplier = "Tape Max Speed Multiplier"
        tapeMaxSpeedSmoothing = "Tape Max Speed Smoothing"
        tapeMinTraceFail = "Tape Min Trace Fail"
        tapeMinTraceWarn = "Tape Min Trace Warn"
        tapeRelMoveThreshold = "Tape Rel Move Threshold"
        tapeShakycamAllowance = "Tape Shakycam Allowance"
        tapeTraceLimitNum = "Tape Trace Limit Num"

    def __init__(self, parent=None):
        super(ConfigDialog, self).__init__(parent)

        settings.load()
        layout = QtGui.QGridLayout()

        # Collider Suite
        self.colliderSuiteAssimilateSize = self.createQLineEditIntValidator(
                str(settings.COLLIDER_SUITE_ASSIMILATE_SIZE),
                guisettings.COLLIDER_SUITE_ASSIMILATE_SIZE_RANGE,
                self.ToolTips.colliderSuiteAssimilateSize)
        self.colliderSuiteOffshoot = self.createQLineEditIntValidator(
                str(settings.COLLIDER_SUITE_OFFSHOOT),
                guisettings.COLLIDER_SUITE_OFFSHOOT_RANGE,
                self.ToolTips.colliderSuiteOffshoot)
        self.colliderSuiteSplitAbs = self.createQLineEditIntValidator(
                str(settings.COLLIDER_SUITE_SPLIT_ABS),
                guisettings.COLLIDER_SUITE_SPLIT_ABS_RANGE,
                self.ToolTips.colliderSuiteSplitAbs)
        self.colliderSuiteSplitRel = self.createQLineEditDoubleValidator(
                str(settings.COLLIDER_SUITE_SPLIT_REL),
                guisettings.COLLIDER_SUITE_SPLIT_REL_RANGE,
                self.ToolTips.colliderSuiteSplitRel)

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
        self.tapeFrameSearchLimit = self.createQLineEditIntValidator(
                str(settings.TAPE_FRAME_SEARCH_LIMIT),
                guisettings.TAPE_FRAME_SEARCH_LIMIT_RANGE,
                self.ToolTips.tapeFrameSearchLimit)
        self.tapeKdeSamples = self.createQLineEditIntValidator(
                str(settings.TAPE_KDE_SAMPLES),
                guisettings.TAPE_KDE_SAMPLES_RANGE,
                self.ToolTips.tapeKdeSamples)
        self.tapeMaxSpeedMultiplier = self.createQLineEditDoubleValidator(
                str(settings.TAPE_MAX_SPEED_MULTIPLIER),
                guisettings.TAPE_MAX_SPEED_MULTIPLIER_RANGE,
                self.ToolTips.tapeMaxSpeedMultiplier)
        self.tapeMaxSpeedSmoothing = self.createQLineEditIntValidator(
                str(settings.TAPE_MAX_SPEED_SMOOTHING),
                guisettings.TAPE_MAX_SPEED_SMOOTHING_RANGE,
                self.ToolTips.tapeMaxSpeedSmoothing)
        self.tapeMinTraceFail = self.createQLineEditIntValidator(
                str(settings.TAPE_MIN_TRACE_FAIL),
                guisettings.TAPE_MIN_TRACE_FAIL_RANGE,
                self.ToolTips.tapeMinTraceFail)
        self.tapeMinTraceWarn = self.createQLineEditIntValidator(
                str(settings.TAPE_MIN_TRACE_WARN),
                guisettings.TAPE_MIN_TRACE_WARN_RANGE,
                self.ToolTips.tapeMinTraceWarn)
        self.tapeRelMoveThreshold = self.createQLineEditDoubleValidator(
                str(settings.TAPE_REL_MOVE_THRESHOLD),
                guisettings.TAPE_REL_MOVE_THRESHOLD_RANGE,
                self.ToolTips.tapeRelMoveThreshold)
        self.tapeShakycamAllowance = self.createQLineEditIntValidator(
                str(settings.TAPE_SHAKYCAM_ALLOWANCE),
                guisettings.TAPE_SHAKYCAM_ALLOWANCE_RANGE,
                self.ToolTips.tapeShakycamAllowance)
        self.tapeTraceLimitNum = self.createQLineEditIntValidator(
                str(settings.TAPE_TRACE_LIMIT_NUM),
                guisettings.TAPE_TRACE_LIMIT_NUM_RANGE,
                self.ToolTips.tapeTraceLimitNum)

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

        # Buttons
        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Save|QtGui.QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.button(QtGui.QDialogButtonBox.Save).clicked.connect(self.save_clicked)
        buttons.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.cancel_clicked)
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addLayout(layout)
        mainLayout.addWidget(buttons)
        self.setLayout(mainLayout)
        self.setFixedSize(self.minimumSize())
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.CustomizeWindowHint)

    def createQLineEditIntValidator(self, value, range, toolTip):
        result = QtGui.QLineEdit(value)
        result.setValidator(QtGui.QIntValidator(range[0], range[1], self))
        result.setToolTip(toolTip)
        result.setMinimumSize(50, result.minimumSize().height())
        return result

    def createQLineEditDoubleValidator(self, value, range, toolTip):
        result = QtGui.QLineEdit(value)
        result.setValidator(QtGui.QDoubleValidator(range[0], range[1], range[2], self))
        result.setToolTip(toolTip)
        result.setMinimumSize(50, result.minimumSize().height())
        return result

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

        settings.save()
        self.close()

    def cancel_clicked(self, ev):
        self.close()
