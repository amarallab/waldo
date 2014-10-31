__author__ = 'heltena'

import time

def silly_function(elapse, callback):
    MAX = 10
    for i in range(MAX):
        time.sleep(elapse)
        value = int(i/float(MAX)*100) / 100.0
        callback(value)
