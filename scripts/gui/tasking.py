__author__ = 'heltena'

from PyQt4 import QtCore


class _WorkerCancelled(Exception):
    pass


class _Worker(QtCore.QObject):
    madeProgress = QtCore.pyqtSignal([float])
    finished = QtCore.pyqtSignal()
    finish = False

    def __init__(self, fnc, **kwargs):
        QtCore.QObject.__init__(self)
        self.fnc = fnc
        self.kwargs = kwargs
        self.prev_value = -1

    def run(self):
        try:
            self.kwargs['callback'] = self._callback
            self.fnc(**self.kwargs)
            self.finished.emit()
        except _WorkerCancelled:
            pass

    def _callback(self, value):
        if self.prev_value != value:
            self.madeProgress.emit(value)
            self.prev_value = value
        if self.finish:
            raise _WorkerCancelled()


class CommandTask:
    def __init__(self, _madeProgress, _finished):
        self.worker = None
        self.thread = None
        self._madeProgress = _madeProgress
        self._finished = _finished

    def start(self, fnc, **kwargs):
        self.worker = _Worker(fnc, **kwargs)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.worker.madeProgress.connect(self._madeProgress)
        self.worker.finished.connect(self._finished)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def requestCancel(self):
        self.worker.finish = True
        self.thread.quit()
        self.thread.wait()
