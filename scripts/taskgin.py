import sys
from six.moves import queue
import threading
import multiprocessing as mp

class TaskGin(object):
    def __init__(self, ftask, faccount=None, n_workers=None):
        """
        Task Gin.

        1. Initialize the object with a callable (*ftask*) that does
           something.
        2. Start the workers (``.start()``).
        3. Call ``.do_task()`` as you would **ftask**.
        4. ``.wait()`` until all tasks have been run.

        Parameters
        ----------
        ftask : callable
            Called

        Keyword Arguments
        -----------------
        faccount : callable
            Called with the result of each call to *ftask*. If None, not used.
        n_workers : int
            Number of processes to use for task running. Defaults to the
            number of CPUs.
        """
        self.ftask = ftask
        self.n_workers = n_workers or mp.cpu_count()

        self.taskq = mp.JoinableQueue(self.n_workers + 2)
        self.resultq = mp.Queue()

        self.workers = [mp.Process(
                name='Worker-{}'.format(n),
                target=self._worker,
                #args=(self.ftask, self.taskq),
            ) for n in range(self.n_workers)]

        if faccount:
            self.faccount = faccount
            self.accountant = threading.Thread(
                name='Accountant',
                target=self._accountant,
            )
        else:
            self.accountant = None

    def _worker(self):
        while True:
            try:
                args, kwargs = self.taskq.get()
                if args is None:
                    self.taskq.task_done()
                    break

                try:
                    result = self.ftask(*args, **kwargs)
                except Exception as e:
                    result = 'args: {}, kwargs: {} -> {}'.format(args, kwargs, repr(e))

                self.taskq.task_done()
                self.resultq.put(result)
            except (KeyboardInterrupt, SystemExit):
                return

    def _accountant(self):
        while True:
            try:

                self.resultq.get(block=block)
            except (KeyboardInterrupt, SystemExit):
                return

    def start(self):
        for w in self.workers:
            w.start()
        if self.accountant:
            self.accountant.start()

    def do_task(self, *args, **kwargs):
        self.taskq.put((args, kwargs))

    def wait(self):
        """
        Signal all workers to shut down then wait for outstanding jobs to
        finish.
        """
        for n in range(self.n_workers):
            self.taskq.put((None, None))
        self.taskq.join()
        if self.accountant:
            self.accountant.join()

    def stop(self):
        """
        Dump all existing tasks in the queue, signal workers to shut down,
        then wait for outstanding jobs to finish.
        """
        try:
            while True:
                self.taskq.get(False)
        except queue.Empty:
            pass
        self.wait()

if __name__ == '__main__':
    # demo/test
    import sys
    import time
    import random

    tasks = [random.random()/3 + 1 for n in range(30)]
    print(tasks)

    def job(x):
        time.sleep(x)
        if x > 1.2:
            raise ValueError('test')
        return -x

    gin = TaskGin(job)

    gin.start()

    results = []

    for task in tasks:
        print('loading task {}'.format(task))
        while True:
            try:
                gin.do_task(task)
                break
            except queue.Full:
                results.append(gin.get_result(block=True))

    gin.wait()

    while True:
        try:
            results.append(gin.get_result())
        except queue.Empty:
            break

    print(results)

