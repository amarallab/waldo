__author__ = 'heltena'

import numpy as np
import profiling
import sys

def peter_add_border(im):
    new_image = np.zeros([len(im) + 2, len(im[0]) + 2], dtype=int)
    for i, im_row in enumerate(im):
        for j, _ in enumerate(im_row):
            new_image[i + 1][j + 1] = im[i][j]
    return new_image

def peter_remove_border(im):
    new_image = np.zeros([len(im) - 2, len(im[0]) - 2], dtype=int)
    for i, im_row in enumerate(im[1:-1], start=1):
        for j, _ in enumerate(im_row[1:-1], start=1):
            new_image[i - 1][j - 1] = im[i][j]
    return new_image

def heltena_add_border(im):
    new_image = np.zeros([len(im) + 2, len(im[0]) + 2], dtype=int)
    new_image[1:len(im)+1,1:len(im[0])+1] = im
    return new_image

def heltena_remove_border(im):
    return im[1:len(im)-1,1:len(im[0])-1]

def run_test(name, im, ab, rb):
    x = ab(im)
    x = rb(x)

    if not np.array_equal(im, x):
        print "E: Not equals on test %s" % name
        sys.exit(-1)

    profiling.begin("Test %s" % name)
    for i in range(1000):
        x = ab(im)
        x = rb(x)
    profiling.end("Test %s" % name)

im = np.zeros([50, 50], dtype=int)
run_test("peter", im, peter_add_border, peter_remove_border)
run_test("heltena", im, heltena_add_border, heltena_remove_border)
