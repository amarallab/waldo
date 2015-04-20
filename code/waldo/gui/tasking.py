from __future__ import absolute_import, print_function

__author__ = 'heltena'

# standard library
import traceback

# third party
import numpy as np
from PyQt4 import QtCore

class _WorkerCancelled(Exception):
    pass


class _Worker(QtCore.QObject):
    madeProgress = QtCore.pyqtSignal([int, float])
    madeNumpyArrayProgress = QtCore.pyqtSignal([int, np.ndarray])
    finished = QtCore.pyqtSignal()
    cancelled = QtCore.pyqtSignal()
    finish = False

    def __init__(self, fnc, **kwargs):
        QtCore.QObject.__init__(self)
        self.fnc = fnc
        self.kwargs = kwargs
        self.prev_values = {}

    def run(self):
        try:
            self.prev_values = {}
            self.kwargs['callback'] = self._callback
            self.fnc(**self.kwargs)
            self.finished.emit()
        except _WorkerCancelled:
            self.cancelled.emit()
        except Exception, ex:
            traceback.print_exc()
            self.cancelled.emit()

    def _callback(self, item, value):
        if type(value) in [int, float]:
            if item not in self.prev_values or self.prev_values[item] != value:
                self.prev_values[item] = value
                self.madeProgress.emit(item, value)
        elif type(value) == np.ndarray:
            self.madeNumpyArrayProgress.emit(item, value)
        else:
            print("E: error in parameters")
        if self.finish:
            raise _WorkerCancelled()


class CommandTask:
    def __init__(self, _madeProgress, _finished, _cancelled, _madeNumpyArrayProgress=None):
        self.worker = None
        self.thread = None
        self._madeProgress = _madeProgress
        self._finished = _finished
        self._cancelled = _cancelled
        self._madeNumpyArrayProgress = _madeNumpyArrayProgress

    def start(self, fnc, **kwargs):
        self.worker = _Worker(fnc, **kwargs)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.worker.madeProgress.connect(self._madeProgress)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.finished.connect(self._finished)
        if self._madeNumpyArrayProgress is not None:
            self.worker.madeNumpyArrayProgress.connect(self._madeNumpyArrayProgress)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def requestCancel(self):
        self.worker.finish = True

    def waitFinished(self):
        self.thread.quit()
        self.thread.wait()
