__author__ = 'heltena'

import time

def silly_function(elapse, callback):
    MAX = 10
    for i in range(MAX):
        time.sleep(elapse)
        value = int(i/float(MAX)*100) / 100.0
        print "TEST value: %.2f" % value
        callback(0, value)


def silly_function_2(task_num, elapse, callback):
    MAX = 10
    for task in range(task_num):
        for i in range(MAX):
            time.sleep(elapse)
            value = int(i/float(MAX)*100) / 100.0
            print "TEST (%d) value: %.2f" % (task, value)
            callback(1, value)
        callback(0, task + 1)
