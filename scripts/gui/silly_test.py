__author__ = 'heltena'

import time

def silly_function(callback):
    MAX = 10
    for i in range(MAX):
        time.sleep(1)
        callback(i/float(MAX))
