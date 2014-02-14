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
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_directory)

# nonstandard imports
#from Shared.Code.wormmetrics.spine_measures import compute_spine_measures
#from Shared.Code.wormmetrics.centroid_measures import compute_centroid_measures
#from Shared.Code.wormmetrics.basic_measures import compute_basic_measures
from Shared.Code.WormMetrics.switchboard import pull_metric_for_blob_id
from Shared.Code.Database.mongo_retrieve import pull_data_type_for_blob
#from Shared.Code.Database.mongo_retrieve import unique_blob_ids_for_query
#from Shared.Code.Database.mongo_insert import compute_and_insert_measurements
#from Shared.Code.Database.mongo_insert import filter_skipped_and_out_of_rang
import Shared.Code.Database.mongo_support_functions as mongo
from Shared.Code.Settings.data_settings import mongo_settings
from Shared.Code.Database.mongo_retrieve import timedict_to_list
from Shared.Code.Analysis.filter_utilities import savitzky_golay

DATA_DIR = './../Data/Single-Speeds/'
'''
def smooth_in_spacetime(t, x, repeated_smoothings=5,
                        space_poly_order=DEFAULT_POLY_ORDER, space_running_window_size=DEFAULT_WINDOW_SIZE,
                        time_poly_order=DEFAULT_POLY_ORDER, time_running_window_size=DEFAULT_WINDOW_SIZE):
    pass
'''                                

def write_speeds_for_blob_id(blob_id, savedir=DATA_DIR, **kwargs):
    """
    """
    savename = '{path}{bID}_speeds.json'.format(path=savedir, bID=blob_id)
    print savename
    cent_timedict = pull_metric_for_blob_id(blob_id, metric='centroid_speed', **kwargs)
    tc, cent = timedict_to_list(cent_timedict)
    along_timedict = pull_metric_for_blob_id(blob_id, metric='speed_along', **kwargs)
    ta, along = timedict_to_list(along_timedict)
    perp_timedict = pull_metric_for_blob_id(blob_id, metric='speed_perp', **kwargs)
    tp, perp = timedict_to_list(perp_timedict)
    xy_timedict = pull_data_type_for_blob(blob_id, data_type='xy_raw', **kwargs)['data']
    t, xy = timedict_to_list(xy_timedict)
    times_and_speeds = tc, cent, ta, along, tp, perp, t ,xy
    json.dump(times_and_speeds, open(savename, 'w'))
    return times_and_speeds

def write_spine_json(blob_id, savedir='./', **kwargs):
    savename = '{path}{bID}_spine.json'.format(path=savedir, bID=blob_id)
    xy_timedict = pull_data_type_for_blob(blob_id, data_type='smoothed_spine', **kwargs)['data']
    t, spines = timedict_to_list(xy_timedict)
    json.dump({'times':t, 'spines':spines}, open(savename, 'w'))


def sample_worms():
    return ["20130423_100249_00317",
            "20130319_161143_01727", "20130326_164206_02219", "20130328_155830_01102",
            "20130905_114409_01981", "20130319_124109_00710",
            "20130416_160141_01392",
            "20130408_155445_01529",
            "20130911_152233_01703",
            "20130415_104153_00853",
            "20130611_151557_03751" ]

def write_samples(blob_ids):
    """
    
    Arguments:
    - `blob_id`:
    """
    try:
        # initialize the connection to the mongo client
        mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                                   mongo_settings['worm_db'], mongo_settings['blob_collection'])
        
        for blob_id in blob_ids:
            try:
                times_and_speeds= write_speeds_for_blob_id(blob_id, mongo_client=mongo_client)
            except Exception as e:
                print blob_id, e
    finally:
        mongo_client.close()

def plot_dist_against_speed_along(blob_id, prange=None, data_dir=DATA_DIR):
    files = glob.glob('{path}/{bID}*speeds.json'.format(path=data_dir, bID=blob_id))
    if len(files) != 1:
        print 'weird, while looking for one file, found:', files
    speed_and_times = json.load(open(files[0], 'r'))
    tc, cent, ta, along, tp, perp, t, xy  = speed_and_times

    
    x, y = zip(*xy)
    dist = [0] 
    d = 0
    for i in range(1, len(xy)):
        dx = (x[i] - x[i-1])
        dy = (y[i] - y[i-1])
        d += math.sqrt(dx**2 + dy**2)
        dist.append(d)

    if not prange:
        prange = [min(tc), max(tc)]

    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(ta, along)
    plt.plot([min(ta), max(ta)], [0, 0], color='black')
    plt.xlim(prange)

    plt.subplot(2,1,2)
    plt.plot(t, dist)
    plt.xlim(prange)

    plt.show()

def plot_xy_vs_speeds(blob_id, prange=None, data_dir=DATA_DIR):
    files = glob.glob('{path}/{bID}*speeds.json'.format(path=data_dir, bID=blob_id))
    if len(files) != 1:
        print 'weird, while looking for one file, found:', files
    speed_and_times = json.load(open(files[0], 'r'))
    tc, cent, ta, along, tp, perp, t, xy  = speed_and_times

    
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
    files = glob.glob('{path}/{bID}*speeds.json'.format(path=data_dir, bID=blob_id))
    if len(files) != 1:
        print 'weird, while looking for one file, found:', files
    speed_and_times = json.load(open(files[0], 'r'))
    tc, cent, ta, along, tp, perp, time, xy  = speed_and_times
    t = ta
    s = along

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
    # read data
    files = glob.glob('{path}/{bID}*speeds.json'.format(path=data_dir, bID=blob_id))
    if len(files) != 1:
        print 'weird, while looking for one file, found:', files
    speed_and_times = json.load(open(files[0], 'r'))
    tc, cent, ta, along, tp, perp, time, xy  = speed_and_times
    x, y = zip(*xy[1:])
    x, y = list(x), list(y)

    # choose speed, itialize 

    t = ta
    s = along
    #t = tc
    #s = cent

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
    #write_samples([blob_id])
    #plot_xy_vs_speeds(blob_id)
    #plot_dist_against_speed_along(blob_id)
    plot_path(blob_id)
    #multifilter_plot(blob_id)
    #write_spine_json(blob_id)
