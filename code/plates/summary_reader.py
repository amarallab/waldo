#!/usr/bin/env 

'''
Filename: summary_reader

Description:
'''

__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'
__author__ = 'Peter B. Winter'

import json
import math
#import pandas as pd
import matplotlib.pyplot as plt
from itertools import izip
DATA_DIR = './../Data/'


def parse_lost_and_found(line):
	return line
def parse_file_ids(line):
    return line

def parse_summary_line(line):

    line = line.strip('\r\n')
    lost_and_found, file_ids, events = None, None, None
    sep = line.split('%%%')
    if len(sep) == 2:
    	line, file_ids = sep
    	#file_ids = parse_file_ids(file_id_string)
    sep = line.split('%%')
    if len(sep) == 2:
        line, lost_and_found_string = sep
        lost_and_found = parse_file_ids(lost_and_found_string)
    sep = line.split('%')
    if len(sep) == 2:
        line, events = sep
    row = line.split()
    return row, events, lost_and_found, file_ids

def read_summary(filename):
    """ """
    print filename
    headers = ['frame', 'time','N', 'N-persist', 'persistance', 'px-per-s',
         	   'rad-per-s', 'len', 'rel-len', 'width', 'rel-width',
               'aspect', 'rel-aspect', 'wiggle', 'size']

    data= {}
    for h in headers:
        data[h] = []

    with open(filename, 'r') as f:
    	lines = f.readlines()
        for line in lines[:]:
            p = parse_summary_line(line)
            row, events, lost_and_found, file_ids = p
            for h,c in izip(headers, row):
                data[h].append(c) 
    return data

def main():
    filename = DATA_DIR + 'N2 A1 V3B bleach3.summary'
    data = read_summary(filename)
    plt.figure()
    #plt.plot(data['time'], data['px-per-s'])
    plt.plot(data['time'], data['persistance'])
    #plt.plot(data['time'], data['N'])
    #plt.plot(data['time'], data['N-persist'])
    plt.show()

if __name__ == '__main__':
    main()
