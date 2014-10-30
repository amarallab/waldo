__author__ = 'heltena'

from PyQt4 import QtCore


class _WorkerCancelled(Exception):
    pass


class _Worker(QtCore.QObject):
    madeProgress = QtCore.pyqtSignal([float])
    finished = QtCore.pyqtSignal()
    finish = False

    def __init__(self, fnc):
        QtCore.QObject.__init__(self)
        self.fnc = fnc

    def run(self):
        try:
            self.fnc(self._callback)
            self.finished.emit()
        except _WorkerCancelled:
            pass

    def _callback(self, value):
        self.madeProgress.emit(value)
        if self.finish:
            raise _WorkerCancelled()


class CommandTask:
    def __init__(self):
        self.worker = None
        self.thread = None

    def start(self, fnc, _madeProgress, _finished):
        self.worker = _Worker(fnc)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.worker.madeProgress.connect(_madeProgress)
        self.worker.finished.connect(_finished)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def requestCancel(self):
        self.worker.finish = True
        self.thread.quit()
        self.thread.wait()
