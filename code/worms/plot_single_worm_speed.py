#!/usr/bin/env python

'''
Filename: plot_single_worm_speed.py
Description: plots the speeds for a one worm at a time.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import json
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_DIR = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared/'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from metrics.measurement_switchboard import pull_blob_data
from wio.file_manager import get_timeseries
from filtering.filter_utilities import savitzky_golay

DATA_DIR = './../Data/Single-Speeds/'

def sample_worms():
    return ["20130423_100249_00317",
            "20130319_161143_01727", "20130326_164206_02219", "20130328_155830_01102",
            "20130905_114409_01981", "20130319_124109_00710",
            "20130416_160141_01392",
            "20130408_155445_01529",
            "20130911_152233_01703",
            "20130415_104153_00853",
            "20130611_151557_03751" ]


def plot_xy_vs_speeds(blob_id, prange=None, data_dir=DATA_DIR):

    tc, cent = pull_blob_data(blob_id, metric='cent_speed')
    #ta, along = pull_blob_data(blob_id, metric='speed_along', **kwargs)
    #tp, perp = pull_blob_data(blob_id, metric='speed_perp', **kwargs)
    t, xy = get_timeseries(blob_id, data_type='xy_raw')


    
    x, y = zip(*xy)
    if not prange:
        prange = [min(tc), max(tc)]

    plt.figure()
    ax1 = plt.subplot(4,1,1)
    plt.plot(tc, cent)
    plt.xlim(prange)

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    plt.plot(ta, along)
    plt.plot([min(ta), max(ta)], [0, 0], color='black')
    plt.xlim(prange)

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    plt.plot(t, x)
    plt.xlim(prange)

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    plt.plot(t, y)
    plt.xlim(prange)

    plt.show()

def multifilter_plot(blob_id, prange=None, data_dir=DATA_DIR):

    t, s = pull_blob_data(blob_id, metric='cent_speed', **kwargs)
    t, xy = get_timeseries(blob_id, data_type='xy_raw')
    if not prange:
        prange = [min(t), max(t)]


    order = 3
    s1 = savitzky_golay(y=np.array(s), window_size=21, order=order)
    s2 = savitzky_golay(y=np.array(s), window_size=51, order=order)
    s3 = savitzky_golay(y=np.array(s), window_size=71, order=order)
    s4 = savitzky_golay(y=np.array(s), window_size=101, order=order)

    '''
    s1 = savitzky_golay(y=np.array(s), window_size=101, order=order)
    s2 = savitzky_golay(y=np.array(s), window_size=201, order=order)
    s3 = savitzky_golay(y=np.array(s), window_size=301, order=order)
    s4 = savitzky_golay(y=np.array(s), window_size=401, order=order)


    s1 = savitzky_golay(y=np.array(s), window_size=101, order=1)
    s2 = savitzky_golay(y=np.array(s), window_size=101, order=2)
    s3 = savitzky_golay(y=np.array(s), window_size=101, order=3)
    s4 = savitzky_golay(y=np.array(s), window_size=101, order=4)
    '''

    plt.figure()
    ax1 = plt.subplot(4,1,1)
    plt.plot(t, s, alpha=0.1, color='black', label='raw')
    plt.plot(t, s1, alpha=0.5, color='blue', label='f')
    plt.plot([min(ta), max(ta)], [0, 0], color='black')
    plt.legend()

    ax2 = plt.subplot(4,1,2, sharex=ax1, sharey=ax1)
    plt.plot(t, s, alpha=0.1, color='black', label='raw')
    plt.plot(t, s2, alpha=0.5, color='blue', label='f')
    plt.plot([min(ta), max(ta)], [0, 0], color='black')
    plt.legend()

    ax3 = plt.subplot(4,1,3, sharex=ax1, sharey=ax1)
    plt.plot(t, s, alpha=0.1,  color='black', label='raw')
    plt.plot(t, s3, alpha=0.5, color='blue', label='f')
    plt.plot([min(ta), max(ta)], [0, 0], color='black')
    plt.legend()

    ax4 = plt.subplot(4,1,4, sharex=ax1, sharey=ax1)    
    plt.plot(t, s, alpha=0.1, color='black', label='raw')
    plt.plot(t, s4, alpha=0.5, color='blue', label='f')
    plt.plot([min(ta), max(ta)], [0, 0], color='black')
    plt.legend()
    plt.xlim(prange)
    plt.show()


def plot_path(blob_id, prange=None, data_dir=DATA_DIR):

    t, s = pull_blob_data(blob_id, metric='cent_speed')
    t, xy = get_timeseries(blob_id, data_type='xy')
    x, y = zip(*xy[1:])
    x, y = list(x), list(y)

    # choose speed, itialize 
    t = tc
    s = cent

    interval = 30
    print 'xy', len(xy)
    print 't', len(t)
    print 's', len(s)
    cmap = cm.jet
    cmap = cm.rainbow
    #cmap = cm.hot

    speed = savitzky_golay(y=np.array(s), window_size=101, order=4)
    max_speed = max(speed)
    norm_speed = map(lambda i: i / max_speed, speed)

    # maked colored segments for speeds
    speedpoints = np.array([t, speed]).T.reshape(-1, 1, 2)
    speedsegments = np.concatenate([speedpoints[:-1], speedpoints[1:]], axis=1)
    lc1 = LineCollection(speedsegments, cmap=cmap) #, norm=norm)
    lc1.set_array(np.array(norm_speed))
    lc1.set_linewidth(1)

    # make colored segments for path
    xypoints = np.array([x, y]).T.reshape(-1, 1, 2)
    xysegments = np.concatenate([xypoints[:-1], xypoints[1:]], axis=1)
    lc2 = LineCollection(xysegments, cmap=cmap) #, norm=norm)
    lc2.set_array(np.array(speed))
    lc2.set_linewidth(2)

    # initialize Graph
    plt.figure(1, figsize=(8,8))
    left, width = 0.1, 0.8
    bottom, height = 0.05, 0.6
    bottom_h = height + 0.1
    label_settings = {'top': 'on', 'bottom': 'on', 'right': 'on', 'left': 'on',
                      'labelbottom': 'off', 'labeltop': 'on', 'labelright': 'off', 'labelleft': 'on'}

    # make speed plot
    speed_ax = plt.axes([left, bottom_h, width, 0.2])
    plt.plot(t, s, label='raw', alpha=0.1)
    plt.plot(t, speed, alpha=0.1)
    plt.plot([t[0], t[-1]], [0, 0], color='black', alpha=0.1)
    plt.gca().add_collection(lc1)

    speed_ax.set_ylim([min(speed), max(speed)])
    speed_ax.set_xlim([tc[0], tc[-1]])    
    plt.tick_params(**label_settings)    
    speed_ax.set_xlabel('Time (s)')
    speed_ax.xaxis.set_label_position('top') 
    speed_ax.set_ylabel('mm/s')

    plt.legend()

    # make path plot
    xy_ax = plt.axes([left, bottom, width, height])
    plt.gca().add_collection(lc2)
    #plt.scatter(x[::], y[::], c=speed[::], cmap=cmap, linewidths=(1,), alpha=0.3, marker=None)
    plt.scatter(x[::interval], y[::interval], c=speed[::interval], cmap=cmap, linewidths=(1,), alpha=0.3, marker=None)
    plt.plot(x[::interval], y[::interval], marker='.', color='black', lw=0, alpha=0.5)
    plt.plot([x[0]], [y[0]], marker='.', color='red')
    plt.axis('equal')
    xy_ax.set_xlabel('x (pix)')
    xy_ax.set_ylabel('y (pix)')
    plt.show()

if __name__ == '__main__':
    #blobs = sample_worms()
    blob_id = '20130415_104153_00853'
    #blob_id = '20130318_131111'
    #plot_xy_vs_speeds(blob_id)
    #plot_dist_against_speed_along(blob_id)
    plot_path(blob_id)
    #multifilter_plot(blob_id)
    #write_spine_json(blob_id)
