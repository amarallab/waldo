#!/usr/bin/env

# standard imports
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
import matplotlib.mlab as mlab
import bisect

import database.mongo_support_functions as mongo
from database.mongo_retrieve import mongo_query
from database.mongo_retrieve import pull_data_type_for_blob
from database.mongo_retrieve import timedict_to_list

from importing.experiment_index import Experiment_Attribute_Index
from settings.local import logistics_settings
from settings.local import mongo_settings

import flag_timepoints
import breaks_and_coils as breaks
from WormMetrics.spine_measures import compute_spine_measures

DATA_DIR = logistics_settings['filesystem_data']
SAVE_DIR = './../Results/Breaks/'


def choose_ex_id(key='purpose', value='N2_aging'):
    ei = Experiment_Attribute_Index()
    return ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value)

def plot_break_elements_for_blobid(blob_id, time_range= [], **kwargs):
    savename = '{d}bid{bid}.png'.format(d=SAVE_DIR, bid=blob_id)
    print savename


    # grab basic flag dict
    all_flag_dicts = pull_data_type_for_blob(blob_id, 'flags', **kwargs)['data']
    flag_timedict = flag_timepoints.consolidate_flags(all_flag_dicts)
    ftimes, flags = timedict_to_list(flag_timedict)
    flag_points = [t for (t, f) in izip(ftimes, flags) if not f]

    # flagdict with loner flags removed
    new_flagdict = breaks.remove_loner_flags(flag_timedict)
    ftimes2, flags2 = timedict_to_list(new_flagdict)
    flag_points2 = [t for (t, f) in izip(ftimes2, flags2) if not f]

    break_list = pull_data_type_for_blob(blob_id, 'breaks', **kwargs)['data'].values()

    #print 'bd', break_dict
    #good_regions = breaks.good_segments_from_timedict(break_dict=break_dict, timedict=flag_timedict)
    #print 'good region', good_regions

    curvature_timedict = compute_spine_measures(blob_id=blob_id, metric='curvature_all', smooth=False, **kwargs)
    ctimes, curves = timedict_to_list(curvature_timedict, remove_skips=True)

    if not time_range:
        time_range = [min(ftimes), max(ftimes)]


    crange = max(curves) - min(curves)
    flag_midline = max(curves) + (crange * 0.3)
    flag_height = crange * 0.06
    break_box_margin = crange * 0.1

    flag_y_range = [flag_midline - flag_height, flag_midline + flag_height]
    break_box_bottom = min([min(curves), flag_y_range[0]]) - break_box_margin
    break_box_top = max([max(curves), flag_y_range[1]]) + break_box_margin
    break_box_ys = [break_box_bottom, break_box_top, break_box_top, break_box_bottom, break_box_bottom]

    fig = plt.figure(1)
    # draw flags
    for p in flag_points:
        plt.plot([p, p], flag_y_range, color='red')

    plt.fill([time_range[0], time_range[1], time_range[1], time_range[0]],
             [flag_y_range[0], flag_y_range[0], flag_y_range[1], flag_y_range[1]],
             color='black', alpha=0.1)
    #plt.plot(time_range, [flag_y_range[1], flag_y_range[1]], color='black')
    # draw break boxes
    for (x1, x2) in break_list:
        x1 = float(x1.replace('?', '.'))
        x2 = float(x2.replace('?', '.'))
        #plt.plot([x1,x1,x2,x2,x1], break_box_ys, color='black')
        plt.fill([x1,x1,x2,x2,x1], break_box_ys, color='red', alpha=0.1)

    # draw
    plt.plot(ctimes, curves, marker='.', lw=0)

    break_positions = [bisect.bisect(ctimes, float(x1.replace('?', '.'))) for (x1, x2) in break_list]
    '''
    for (b, br) in zip(break_positions, break_list):
        print b, br
    for (b, br) in zip(break_positions, break_list):
        print ctimes[b], br
    '''
    last_b = 0
    for b in break_positions:
        plt.plot(ctimes[last_b:b], curves[last_b:b], alpha=0.3, color='blue')
        last_b = b
    plt.plot(ctimes[last_b:-1], curves[last_b:-1], alpha=0.3, color='blue')

    plt.tick_params(top='off', bottom='on', right='off', left='off',
                   labelbottom='on', labeltop='off', labelright='off', labelleft='off')
    plt.xlim(time_range)
    plt.xlabel('time (s)')
    plt.ylabel('avg curvature (mm^-1)')
    plt.show()


def plot_break_elements_for_blobid1(blob_id, time_range= [], **kwargs):
    savename = '{d}bid{bid}.png'.format(d=SAVE_DIR, bid=blob_id)
    print savename


    # grab basic flag dict
    all_flag_dicts = pull_data_type_for_blob(blob_id, 'flags', **kwargs)['data']
    flag_timedict = flag_timepoints.consolidate_flags(all_flag_dicts)
    ftimes, flags = timedict_to_list(flag_timedict)
    flag_points = [t for (t, f) in izip(ftimes, flags) if not f]

    # flagdict with loner flags removed
    new_flagdict = breaks.remove_loner_flags(flag_timedict)
    ftimes2, flags2 = timedict_to_list(new_flagdict)
    flag_points2 = [t for (t, f) in izip(ftimes2, flags2) if not f]

    break_list = pull_data_type_for_blob(blob_id, 'breaks', **kwargs)['data'].values()

    #print 'bd', break_dict
    #good_regions = breaks.good_segments_from_timedict(break_dict=break_dict, timedict=flag_timedict)
    #print 'good region', good_regions

    curvature_timedict = compute_spine_measures(blob_id=blob_id, metric='curvature_all', smooth=False, **kwargs)
    ctimes, curves = timedict_to_list(curvature_timedict, remove_skips=True)

    if not time_range:
        time_range = [min(ftimes), max(ftimes)]

    plt.figure(1)
    ax = plt.subplot(3, 1, 1)
    for p in flag_points:
        ax.plot([p, p], [0,1], color='red')
        ax.set_xlim(time_range)
        ax.set_ylim([-0.2, 1.2])
        ax.tick_params(top='on', bottom='off', right='off', left='off',
                       labelbottom='off', labeltop='off', labelright='off', labelleft='off')

    ax = plt.subplot(3, 1, 2)
    for p in flag_points2:
        ax.plot([p, p], [0,1], color='red')
        ax.set_xlim(time_range)
        ax.set_ylim([-0.2, 1.2])
        ax.tick_params(top='off', bottom='off', right='off', left='off',
               labelbottom='off', labeltop='off', labelright='off', labelleft='off')

    for (x1, x2) in break_list:
        x1 = float(x1.replace('?', '.'))
        x2 = float(x2.replace('?', '.'))
        ax.set_xlim(time_range)
        ax.set_ylim([-0.2, 1.2])
        ax.plot([x1,x1,x2,x2,x1], [-.1, 1.1, 1.1, -.1, -.1], color='black')
        ax.fill([x1,x1,x2,x2,x1], [-.1, 1.1, 1.1, -.1, -.1], color='red', alpha=0.1)

    ax = plt.subplot(3, 1, 3)
    ax.plot(ctimes, curves, marker='.', lw=0)
    for (x1, x2) in break_list:
        x1 = float(x1.replace('?', '.'))
        x2 = float(x2.replace('?', '.'))
        ax.set_xlim(time_range)
        ax.set_ylim([-0.2, 1.2])
        ax.plot([x1,x1,x2,x2,x1], [-.1, 1.1, 1.1, -.1, -.1], color='black')
        ax.fill([x1,x1,x2,x2,x1], [-.1, 1.1, 1.1, -.1, -.1], color='red', alpha=0.1)
    ax.tick_params(top='off', bottom='on', right='off', left='off',
                   labelbottom='off', labeltop='off', labelright='off', labelleft='off')
    ax.set_xlim(time_range)

    plt.show()


def main():
    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['blob_collection'])
    try:
        ex_ids = choose_ex_id()
        for ex_id in ex_ids[5:6]:
            print ex_id
            blobs = [e['blob_id'] for e in mongo_query({'ex_id': ex_id, 'data_type': 'smoothed_spine'}, {'blob_id':1})]
            for blob_id in blobs:
                print blob_id
                plot_break_elements_for_blobid(blob_id, mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)


def plot_one_blob():
    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['blob_collection'])

    try:
        #blob_id = '20130318_153742_02796'
        blob_id = '20130320_102312_10828'
        blob_id = '20130320_102312_19189'
        savename = '{d}bid{bid}.png'.format(d=SAVE_DIR, bid=blob_id)
        plot_break_elements_for_blobid(blob_id, mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)

if __name__ == '__main__':
    plot_one_blob()
    #main()
