import sys
from six.moves import queue
import threading
import multiprocessing as mp

class TaskGin(object):
    def __init__(self, ftask, faccount=None, n_workers=None):
        """
        Task Gin.

        1. Initialize the object with a callable (*ftask*) that does
           something you want.
        2. Start the workers (``.start()``).
        3. Call ``.do_task()`` as you would *ftask*. It will block if the
           queue is full.
        4. ``.wait()`` until all tasks finish running.

        Parameters
        ----------
        ftask : callable
            Your task to run in parallel.

        Keyword Arguments
        -----------------
        faccount : callable
            Called with the args, kwargs, and result of each call to *ftask*
            in a 3-ple. Should be much quicker than the task call. If
            ``None``, the return value is just ignored.
        n_workers : int
            Number of processes to use for task running. Defaults to the
            number of CPUs.
        """
        self.ftask = ftask
        self.n_workers = n_workers or mp.cpu_count()

        self.taskq = mp.JoinableQueue(self.n_workers + 2)

        self.workers = [mp.Process(
                name='Worker-{}'.format(n),
                target=self._worker,
                #args=(self.ftask, self.taskq),
            ) for n in range(self.n_workers)]

        if faccount:
            self.done = threading.Event()
            self.faccount = faccount
            self.accountant = threading.Thread(
                name='Accountant',
                target=self._accountant,
            )
            self.resultq = mp.Queue()
        else:
            self.accountant = None

    def _worker(self):
        while True:
            try:
                task = self.taskq.get()
                if task is None:
                    self.taskq.task_done()
                    return
                args, kwargs = task

                try:
                    result = self.ftask(*args, **kwargs)
                except Exception as e:
                    result = e

                if self.resultq:
                    self.resultq.put((args, kwargs, result))

                self.taskq.task_done()
            except (KeyboardInterrupt, SystemExit):
                return

    def _accountant(self):
        try:
            while not self.done.is_set():
                try:
                    self.faccount(self.resultq.get(True, 0.100))
                except queue.Empty:
                    pass
        except (KeyboardInterrupt, SystemExit):
            return

    def start(self):
        for w in self.workers:
            w.start()
        if self.accountant:
            self.accountant.start()

    def do_task(self, *args, **kwargs):
        try:
            self.taskq.put((args, kwargs))
        except (KeyboardInterrupt, SystemExit):
            self.done.set()
            raise

    def wait(self):
        """
        Wait for outstanding jobs to finish.
        """
        for n in range(self.n_workers):
            try:
                self.taskq.put(None)
            except (KeyboardInterrupt, SystemExit):
                self.done.set()
                raise
        self.taskq.join()

        if self.accountant:
            self.done.set()
            self.accountant.join()

if __name__ == '__main__':
    # demo/test
    import sys
    import time
    import random

    tasks = [random.random()*10 + 1 for n in range(100)]
    print(tasks)

    def job(x):
        time.sleep(x)
        if x > 10:
            raise ValueError('test')
        return -x

    results = []
    def account(x):
        results.append(x)
        print(x)

    gin = TaskGin(job, account)

    gin.start()

    for task in tasks:
        print('loading task {}'.format(task))
        gin.do_task(x=task)

    gin.wait()

    print(results)

