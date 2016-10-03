from __future__ import absolute_import

__author__ = 'heltena'

from waldo.wio import Experiment
import enum

class WaldoBatchRunResult(enum.Enum):
    CACHED = 0
    SUCCEEDED = 1
    FAILED = 2


class WaldoAppData:
    def __init__(self):
        self.single_mode = True
        # Single mode data
        self.selected_ex_id = None
        self.experiment = None
        self.threshold = 0
        self.roi_center = (0, 0)
        self.roi_radius = 0
        self.single_result_message = (WaldoBatchRunResult.CACHED, None)  # (WaldoBatchRunResult, (exception, traceback)|None)

        # Batch mode data
        self.experiment_id_list = []
        self.no_thresholdcache_experiment_id_list = []
        self.batch_result_messages = {}  # (WaldoBatchRunResult, (exception, traceback)|None)

    def singleMode(self):
        self.single_mode = True
        self.single_result_message = (WaldoBatchRunResult.CACHED, None)
        self.batch_result_messages = {}

    def batchMode(self):
        self.single_mode = False
        self.single_result_message = (WaldoBatchRunResult.CACHED, None)
        self.batch_result_messages = {}

    def loadSelectedExperiment(self):
        if not self.single_mode:
            self.experiment = None
            return

        if self.selected_ex_id is not None:
            self.loadExperiment(self.selected_ex_id)
        else:
            self.experiment = None

    def loadExperiment(self, ex_id):
        self.experiment = Experiment(experiment_id=ex_id)
