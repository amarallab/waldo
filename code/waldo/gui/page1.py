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

        self.qDataLabel = QtGui.QLabel(settings.QUALITY_REPORT_ROOT)
        qualityDataButton = QtGui.QPushButton("Change")
        qualityDataButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        qualityDataButton.clicked.connect(self.qualityDataButton_clicked)

        row = 0
        folderLayout.addWidget(QtGui.QLabel("<b>Folders</b>"), row, 0, 1, 3)

        row += 1
        folderLayout.addWidget(QtGui.QLabel("Raw Data"), row, 0, 1, 1)
        folderLayout.addWidget(self.rawDataLabel, row, 1, 1, 1)
        folderLayout.addWidget(rawDataButton, row, 2, 1, 1)

        row += 1
        folderLayout.addWidget(QtGui.QLabel("Project Data"), row, 0, 1, 1)
        folderLayout.addWidget(self.waldoDataLabel, row, 1, 1, 1)
        folderLayout.addWidget(waldoDataButton, row, 2, 1, 1)

        row += 1
        folderLayout.addWidget(QtGui.QLabel("Quality Reports"), row, 0, 1, 1)
        folderLayout.addWidget(self.qDataLabel, row, 1, 1, 1)
        folderLayout.addWidget(qualityDataButton, row, 2, 1, 1)

        self.runBatchModeCheckBox = QtGui.QCheckBox("Run in Batch Mode")
        self.runBatchModeCheckBox.setChecked(self.data.experiment_id_list is not None)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(config_button)
        layout.addLayout(folderLayout)
        layout.addWidget(self.runBatchModeCheckBox)
        self.setLayout(layout)

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

    # PBW 
    def qualityDataButton_clicked(self, ev):
        result = str(QtGui.QFileDialog.getExistingDirectory(directory=self.qDataLabel.text()))
        if len(result) > 0:
            self.qDataLabel.setText(result)
            settings.QUALITY_REPORT_ROOT = str(self.qDataLabel.text())
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
        tapePixelSearchLimit = "Tape Pixel Search Limit"
        defaultCalibrationEnclosureSize = "Default Calibration Enclosure Size"
        collisionPixelOverlapMargin = "Collision Pixel Overlap"

        roiBorderOffset = "Pixels to contract the borders on guess ROI polygon"
        roiCornerOffset = "Pixels to contract in the corners on guess ROI polygon"

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

        row = 0
        # text, row, column, row_height, row_width
        # Title
        layout.addWidget(QtGui.QLabel("<b>Cleaning</b>"), row, 0, 1, 2)

        row += 1
        layout.addWidget(QtGui.QLabel("Assimilate size"), row, 0, 1, 1) # label
        layout.addWidget(self.colliderSuiteAssimilateSize, row, 1, 1, 1) # text box

        row += 1
        layout.addWidget(QtGui.QLabel("Offshoot"), row, 0, 1, 1)
        layout.addWidget(self.colliderSuiteOffshoot, row, 1, 1, 1)

        row += 1
        layout.addWidget(QtGui.QLabel("Split Abs"), row, 0, 1, 1)
        layout.addWidget(self.colliderSuiteSplitAbs, row, 1, 1, 1)

        row += 1
        layout.addWidget(QtGui.QLabel("Split Rel"), row, 0, 1, 1)
        layout.addWidget(self.colliderSuiteSplitRel, row, 1, 1, 1)

        # Tape
        # store text box contents as appropriate variables
        self.tapeFrameSearchLimit = self.createQLineEditIntValidator(
            str(settings.TAPE_FRAME_SEARCH_LIMIT),
            guisettings.TAPE_FRAME_SEARCH_LIMIT_RANGE,
            self.ToolTips.tapeFrameSearchLimit)


        self.tapePixelSearchLimit = self.createQLineEditIntValidator(
            str(settings.TAPE_PIXEL_SEARCH_LIMIT),
            guisettings.TAPE_PIXEL_SEARCH_LIMIT_RANGE,
            self.ToolTips.tapePixelSearchLimit)

        self.defaultCalibrationEnclosureSize = self.createQLineEditIntValidator(
            str(settings.DEFAULT_CALIBRATION_ENCLOSURE_SIZE),
            guisettings.DEFAULT_CALIBRATION_ENCLOSURE_SIZE_RANGE,
            self.ToolTips.defaultCalibrationEnclosureSize)
        
        row = 0
        layout.addWidget(QtGui.QLabel("<b>Infer Arcs</b>"), row, 2, 1, 2)

        row += 1
        layout.addWidget(QtGui.QLabel("dt (frames)"), row, 2, 1, 1)
        layout.addWidget(self.tapeFrameSearchLimit, row, 3, 1, 1)

        row += 1
        layout.addWidget(QtGui.QLabel("dD (pixels)"), row, 2, 1, 1)
        layout.addWidget(self.tapePixelSearchLimit, row, 3, 1, 1)

        row += 1
        layout.addWidget(QtGui.QLabel("<b>Calibration</b>"), row, 2, 1, 2)
        
        row += 1
        layout.addWidget(QtGui.QLabel("Default Enclosure Size (mm)"), row, 2, 1, 1)
        layout.addWidget(self.defaultCalibrationEnclosureSize, row, 3, 1, 1)

        #### COLLISION SETTINGS

        self.collisionPixelOverlapMargin = self.createQLineEditIntValidator(
            str(settings.COLLISION_PIXEL_OVERLAP_MARGIN),
            guisettings.COLLISION_PIXEL_OVERLAP_MARGIN_RANGE,
            self.ToolTips.collisionPixelOverlapMargin)

        self.roiBorderOffset = self.createQLineEditIntValidator(
            str(settings.ROI_BORDER_OFFSET),
            guisettings.ROI_BORDER_OFFSET_RANGE,
            self.ToolTips.roiBorderOffset)
        
        self.roiCornerOffset = self.createQLineEditIntValidator(
            str(settings.ROI_CORNER_OFFSET),
            guisettings.ROI_CORNER_OFFSET_RANGE,
            self.ToolTips.roiCornerOffset)
        
        row = 0
        layout.addWidget(QtGui.QLabel("<b>Collision</b>"), row, 4, 1, 2)

        row += 1
        layout.addWidget(QtGui.QLabel("Min Pixel Overlap Diff (pixels)"), row, 4, 1, 1)
        layout.addWidget(self.collisionPixelOverlapMargin, row, 5, 1, 1)

        row += 1
        layout.addWidget(QtGui.QLabel("<b>Guess ROI Polygon</b>"), row, 4, 1, 2)

        row += 1
        layout.addWidget(QtGui.QLabel("Border Offset"), row, 4, 1, 1)
        layout.addWidget(self.roiBorderOffset, row, 5, 1, 1)

        row += 1
        layout.addWidget(QtGui.QLabel("Corner Offset"), row, 4, 1, 1)
        layout.addWidget(self.roiCornerOffset, row, 5, 1, 1)
        
        # Buttons
        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Save | QtGui.QDialogButtonBox.Cancel, Qt.Horizontal,
                                         self)
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
        settings.COLLIDER_SUITE_ASSIMILATE_SIZE = self._intValueOf(self.colliderSuiteAssimilateSize,
                                                                   settings.COLLIDER_SUITE_ASSIMILATE_SIZE)
        settings.COLLIDER_SUITE_OFFSHOOT = self._intValueOf(self.colliderSuiteOffshoot,
                                                            settings.COLLIDER_SUITE_OFFSHOOT)
        settings.COLLIDER_SUITE_SPLIT_ABS = self._intValueOf(self.colliderSuiteSplitAbs,
                                                             settings.COLLIDER_SUITE_SPLIT_ABS)
        settings.COLLIDER_SUITE_SPLIT_REL = self._doubleValueOf(self.colliderSuiteSplitRel,
                                                                settings.COLLIDER_SUITE_SPLIT_REL)

        # settings.TAPE_REL_MOVE_THRESHOLD = self._doubleValueOf(self.tapeRelMoveThreshold,
        #                                                        settings.TAPE_REL_MOVE_THRESHOLD)
        # settings.TAPE_MIN_TRACE_FAIL = self._intValueOf(self.tapeMinTraceFail, settings.TAPE_MIN_TRACE_FAIL)
        settings.TAPE_FRAME_SEARCH_LIMIT = self._intValueOf(self.tapeFrameSearchLimit, settings.TAPE_FRAME_SEARCH_LIMIT)
        settings.TAPE_PIXEL_SEARCH_LIMIT = self._intValueOf(self.tapePixelSearchLimit, settings.TAPE_PIXEL_SEARCH_LIMIT)
        # settings.TAPE_MAX_SPEED_MULTIPLIER = self._doubleValueOf(self.tapeMaxSpeedMultiplier,
        #                                                          settings.TAPE_MAX_SPEED_MULTIPLIER)
        # settings.TAPE_SHAKYCAM_ALLOWANCE = self._intValueOf(self.tapeShakycamAllowance,
        #                                                     settings.TAPE_SHAKYCAM_ALLOWANCE)

        settings.DEFAULT_CALIBRATION_ENCLOSURE_SIZE = self._intValueOf(self.defaultCalibrationEnclosureSize, 
                                                                       settings.DEFAULT_CALIBRATION_ENCLOSURE_SIZE)

        settings.ROI_BORDER_OFFSET = self._intValueOf(self.roiBorderOffset, settings.ROI_BORDER_OFFSET)
        settings.ROI_CORNER_OFFSET = self._intValueOf(self.roiCornerOffset, settings.ROI_CORNER_OFFSET)
    
        settings.save()
        self.close()

    def cancel_clicked(self, ev):
        self.close()
