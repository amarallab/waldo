#!/usr/bin/env python
'''
Filename: evaluate_acuracy.py

Description: holds many low-level scripts for finding, sorting, and saving files
in a rigid directory structure.
'''
from __future__ import absolute_import, print_function
# standard library

# third party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# projects specific
#import waldo.wio as wio

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# def grab_files(ex_id):
#     print(ex_id)
#     prep_data = wio.file_manager.PrepData(ex_id)
#     matches = prep_data.load('matches')
#     base_accuracy = prep_data.load('base_accuracy')
#     # do I need to set index col to 'frame'?
#     #base_accuracy = pd.read_csv(s2, index_col='frame')
#     return base_accuracy, matches

def recalculate_accuracy(matches, base_accuracy, bids=[], unforgiving=False):

    #frames, true_pos, false_pos, false_neg = [], [], [], []
    data = {}
    #print(base_accuracy.head())
    ba_len1= len(base_accuracy)
    base_accuracy.drop_duplicates(cols='frame', inplace=True)
    ba_len2 = len(base_accuracy)
    if ba_len1 != ba_len2:
        print('WARNING: base accuracy file has duplicate frames')
    ba = base_accuracy.set_index('frame')
    for frame, df in matches.groupby('frame'):
        tp_bids = list(df[df['good']]['bid'])
        # false positives are inside roi and not good.
        check = (df['good']== False) & (df['roi']== True)
        fp_bids = list(df[check]['bid'])

        row = ba.loc[frame]
        #print(row)
        base_tp = row['true-pos']
        base_fp = row['false-pos']
        base_fn = int(row['false-neg'])

        #print(len(tp_bids), len(fp_bids))

        if unforgiving:
            #print('in roi', (df['roi'] == True).sum())

            #print('fp', base_fp, row['false-pos'])
            #print('tp', base_tp)
            #print('fn', base_fn)
            assert (df['roi'] == True).sum() == base_fp + base_tp
            ##print(base_fp, len(fp_bids))
            print('fp', len(fp_bids))
            print(fp_bids)
            #assert base_fp == len(fp_bids)

        if len(bids) > 0:
            n_tracked_bids = len([b for b in tp_bids if b in bids])
            n_filtered_bids = len(tp_bids) - n_tracked_bids
            n_fp = len([b for b in fp_bids if b in bids])
        else:
            n_tracked_bids =  len(tp_bids)
            n_filtered_bids = 0
            n_fp = len(fp_bids)

        if unforgiving:
            assert base_tp == n_tracked_bids + n_filtered_bids


        data[frame] = {'true-pos': n_tracked_bids,
                       'false-pos': n_fp,
                       'false-neg': int(base_fn + n_filtered_bids)}

    accuracy = pd.DataFrame(data).T

    return accuracy

def plot_accuracy_time(df, title=''):
    fig, ax = plt.subplots()
    plt.plot(ax, df.index, df['true-pos'], label='true positives')
    plt.plot(ax, df.index, df['false-pos'], label='false pos')
    plt.plot(ax, df.index, df['false-neg'], label='false neg')
    if title:
        ax.set_title(title)
    ax.set_xlabel('frames')
    ax.set_ylabel('Number of Worms')
    ax.legend()
    plt.show()

def plot_accuracy_bar(df, title=''):
    fig, ax = plt.subplots()

    data = [df['true-pos'].mean(),
            df['false-pos'].mean(),
            df['false-neg'].mean()]

    bars = [df['true-pos'].std(),
            df['false-pos'].std(),
            df['false-neg'].std()]
    #print(data)
    ymax = max([data[i] + bars[i] for i in [0,1,2]])

    coverage = float(data[0]) / float(data[0] + data[2])
    print(int(100 * coverage), '% coverage')

    truth = float(data[0]) / float(data[0] + data[1])
    print(int(100 * truth), '% true worms')

    n = 3
    plt.bar(ax, np.arange(n), data, yerr = bars,
            xticklabels=['TP', 'FP', 'FN'], grid='y')

    if title:
        ax.set_title(title)
    #ax.set_xlabel('frames')
    ax.set_ylabel('Number of Worms')
    ax.set_ylim([0, ymax])
    #ax.legend()
    plt.show()
