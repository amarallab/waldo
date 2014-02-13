#!/usr/bin/env

# standard imports
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
import matplotlib.mlab as mlab
import random

import database.mongo_support_functions as mongo
from database.mongo_retrieve import mongo_query
from database.mongo_retrieve import pull_data_type_for_blob
from database.mongo_retrieve import timedict_to_list

from experiment_index import Experiment_Attribute_Index
from settings.local import LOGISTICS as logistics_settings
from settings.local import MONGO as mongo_settings

from Plotting.SingleWorms.single_worm_suite import plot_outline_shapes
from Encoding.decode_outline import pull_outline
import flag_timepoints

DATA_DIR = logistics_settings['filesystem_data']
SAVE_DIR = './../Results/GoodShape-BadShape/'

def choose_ex_id(key='purpose', value='N2_aging'):
    ei = Experiment_Attribute_Index()
    return ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value)

def grab_good_and_bad_shapes_for_blob(blob_id, **kwargs):
    good_shapes, bad_shapes = [], []
    outline_timedict = pull_outline(blob_id, **kwargs)
    times, outlines = timedict_to_list(outline_timedict)
    all_flag_dicts = pull_data_type_for_blob(blob_id, 'flags', **kwargs)['data']
    flag_timedict = flag_timepoints.consolidate_flags(all_flag_dicts)
    times, flags = timedict_to_list(flag_timedict)
    for flag, outline in izip(flags, outlines):
        if flag:
            good_shapes.append(outline)
        else:
            bad_shapes.append(outline)

    print 'good', len(good_shapes)
    print 'bad', len(bad_shapes)

    random.shuffle(good_shapes)
    random.shuffle(bad_shapes)
    return good_shapes, bad_shapes


def grab_spread(blob_id, **kwargs):
    length_timedict = pull_data_type_for_blob(blob_id, 'length', **kwargs)['data']
    _, lengths = timedict_to_list(length_timedict)
    return np.mean(lengths) * 1.2

def draw_good_and_bad_shapes(good_shapes, bad_shapes, savename='', dimensions=[4, 4], spread=70):

    num_shapes = dimensions[0] * dimensions[1]
    good_shapes = good_shapes[:num_shapes]
    bad_shapes = bad_shapes[:num_shapes]

    label_settings = {'top': 'off', 'bottom': 'off', 'right': 'off', 'left': 'off',
                      'labelbottom': 'off', 'labeltop': 'off', 'labelright': 'off', 'labelleft': 'off'}



    plt.figure(1)
    ax = plt.subplot(1, 2, 1)
    plt.tick_params(**label_settings)
    for i, shape in enumerate(good_shapes):
        x_, y_ = zip(*shape)
        x_shift = np.mean(x_) + (i % dimensions[1]) * spread
        y_shift = np.mean(y_) + (i / dimensions[1]) * spread
        x = [(j - x_shift) for j in x_]
        y = [(j - y_shift) for j in y_]
        ax.plot(x, y, color='black')
        ax.fill(x, y, color='blue', alpha=0.3)
        ax.axis('equal')
    ax = plt.subplot(1, 2, 2)
    plt.tick_params(**label_settings)
    for i, shape in enumerate(bad_shapes):
        x_, y_ = zip(*shape)
        x_shift = np.mean(x_) + (i % dimensions[1]) * spread
        y_shift = np.mean(y_) + (i / dimensions[1]) * spread
        x = [(j - x_shift) for j in x_]
        y = [(j - y_shift) for j in y_]
        ax.plot(x, y, color='black')
        ax.fill(x, y, color='red', alpha=0.3)
        ax.axis('equal')
    #plt.show()
    plt.savefig(savename)
    plt.clf()

def main():
    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['blob_collection'])
    try:
        ex_ids = choose_ex_id()
        for ex_id in ex_ids[5:10]:
            print ex_id
            blobs = [e['blob_id'] for e in mongo_query({'ex_id': ex_id, 'data_type': 'smoothed_spine'}, {'blob_id':1})]
            for blob_id in blobs:
                savename = '{d}bid{bid}.png'.format(d=SAVE_DIR, bid=blob_id)
                print savename
                good_shapes, bad_shapes = grab_good_and_bad_shapes_for_blob(blob_id, mongo_client=mongo_client)
                draw_good_and_bad_shapes(good_shapes, bad_shapes, savename=savename)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)


def plot_one_blob():
    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['blob_collection'])

    try:

        #blob_id = '20130318_153742_02796'
        blob_id = '20130320_102312_10828'
        blob_id = '20130320_102312_19189'
        #image_times = create_image_directory('20130318_153742')
        good_shapes, bad_shapes = grab_good_and_bad_shapes_for_blob(blob_id, mongo_client=mongo_client)
        spread = grab_spread(blob_id, mongo_client=mongo_client)
        draw_good_and_bad_shapes(good_shapes, bad_shapes, spread=spread)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)
if __name__ == '__main__':

    #plot_one_blob()
    main()
