#!/usr/bin/env

# standard library
import math
import bisect

# third party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# waldo
from conf import settings

import database.mongo_support_functions as mongo
from database.mongo_retrieve import mongo_query
from database.mongo_retrieve import pull_data_type_for_blob
from database.mongo_retrieve import timedict_to_list
from ImageManipulation.find_outlines import find_images, crop_image_around_worm

from experiment_index import Experiment_Attribute_Index
from skeletonize_outline import compute_skeleton_from_outline
from spine_processing_videos import process_outlines
from create_spine import smooth_and_space_xy_points


DATA_DIR = settings.LOGISTICS['filesystem_data']
SAVE_DIR = './../Results/BlobShapes/'

def choose_ex_id(key='purpose', value='N2_aging'):
    ei = Experiment_Attribute_Index()
    return ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value)


def plot_shape_progression(blob_id, image_times, **kwargs):
    outline_document = pull_data_type_for_blob(blob_id, 'encoded_outline', **kwargs)
    outline_timedict = outline_document['data']
    spine_timedict = pull_data_type_for_blob(blob_id, 'smoothed_spine', **kwargs)['data']
    start_time, stop_time = outline_document['start_time'], outline_document['stop_time']
    print start_time, stop_time
    useful_pics = {}
    for time in image_times:
        if start_time <= float(time) <= stop_time:
            useful_pics[time] = image_times[time]

    for time in useful_pics.keys()[:]:
        savename = '{d}bid{bid}_time{time}.png'.format(d=SAVE_DIR, bid=blob_id, time=time)
        #savename = ''
        times, outlines = timedict_to_list(outline_timedict)
        i = bisect.bisect(times, float(time)) - 1
        time_key = ('%.3f' %float(times[i])).replace('.', '?')

        raw_outline_timedict, poly_outline_timedict = process_outlines(outline_timedict)
        print time_key, time
        #time = random.choice(raw_outline_timedict.keys())
        raw_outline = raw_outline_timedict[time_key]
        #intermediate_steps = compute_skeleton_from_outline(raw_outline, return_intermediate_steps=True)
        (outline_matrix, filled_matrix, spine_matrix_branched, spine_matrix, xy_shift,
         endpoints) = compute_skeleton_from_outline(raw_outline, return_intermediate_steps=True)#intermediate_steps

        xy_size = (len(outline_matrix), len(outline_matrix[0]))
        cropped_region = crop_image_around_worm(image_file=useful_pics[time], xy_shift=xy_shift, xy_size=xy_size)
        stimes, spines = timedict_to_list(spine_timedict,remove_skips=True)

        i = bisect.bisect(stimes, float(time)) - 1
        if math.fabs(stimes[i] - float(time)) < 0.5:
            spine = spines[i]
            spine_smoothed = True
        else:
            spine = smooth_and_space_xy_points(compute_skeleton_from_outline(raw_outline))
            spine_smoothed = False
        filled_matrix = np.array(filled_matrix)
        spine_matrix_branched = np.array(spine_matrix_branched)
        spine_matrix = np.array(spine_matrix)
        shifted_spine = [(xy[0] - xy_shift[0], xy[1] - xy_shift[1]) for xy in spine]
        plot_single_timepoint(cropped_region, filled_matrix, spine_matrix_branched, spine_matrix,
                              shifted_spine, spine_smoothed, savename)


def plot_single_timepoint(cropped_region, filled_matrix, spine_matrix_branched, spine_matrix, spine, spine_smoothed, savename=None):
    plt.figure(1)
    label_settings = {'top': 'off', 'bottom': 'off', 'right': 'off', 'left': 'off',
                      'labelbottom': 'off', 'labeltop': 'off', 'labelright': 'off', 'labelleft': 'off'}

    # Draw something first, should be an image.
    ax = plt.subplot(1, 4, 1)
    plt.tick_params(**label_settings)
    plt.imshow(cropped_region, cmap=cm.Greys_r)
    #plt.plot(*zip(*raw_outline), color='red')
    ax.set_aspect('equal')

    # Branched outline view
    plt.subplot(1, 4, 2)
    plt.imshow(filled_matrix, interpolation='nearest', alpha=0.3, cmap=cm.Greys_r)
    plt.imshow(spine_matrix_branched, interpolation='nearest', alpha=0.3, cmap=cm.Greys_r)
    plt.tick_params(**label_settings)

    # branches cut view
    plt.subplot(1, 4, 3)
    plt.imshow(filled_matrix, interpolation='nearest', alpha=0.3, cmap=cm.Greys_r)
    plt.imshow(spine_matrix, interpolation='nearest', alpha=0.3, cmap=cm.Greys_r)
    plt.tick_params(**label_settings)

    # final smoothed shape
    ax = plt.subplot(1, 4, 4)
    sx, sy = zip(*spine)

    plt.imshow(cropped_region, cmap=cm.Greys_r)
    if spine_smoothed:
        plt.plot(sy, sx, color='red', lw=1.5)
    else:
        plt.plot(sy, sx, color='yellow', lw=1.5)
    plt.tick_params(**label_settings)
    ax.set_aspect('equal')
    ax.set_xlim([0, len(cropped_region[0])])
    ax.set_ylim([0, len(cropped_region)])
    ax.invert_yaxis()
    if savename:
        plt.savefig(savename)
    else:
        plt.show()
    plt.clf()


def main():
    mongo_client, _ = mongo.start_mongo_client(settings.MONGO['mongo_ip'], settings.MONGO['mongo_port'],
                                               settings.MONGO['worm_db'], settings.MONGO['blob_collection'])
    try:
        ex_ids = choose_ex_id()
        for ex_id in ex_ids[5:10]:
            print ex_id
            image_times = find_images(ex_id)
            blobs = [e['blob_id'] for e in mongo_query({'ex_id': ex_id, 'data_type': 'smoothed_spine'}, {'blob_id':1})]
            for blob_id in blobs:
                print blob_id
                plot_shape_progression(blob_id, image_times, mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)


def plot_one_blob():
    mongo_client, _ = mongo.start_mongo_client(settings.MONGO['mongo_ip'], settings.MONGO['mongo_port'],
                                               settings.MONGO['worm_db'], settings.MONGO['blob_collection'])

    try:
        #blob_id = '20130318_153742_02796'
        blob_id = '20130320_102312_10828'
        blob_id = '20130320_102312_19189'
        #image_times = create_image_directory('20130318_153742')
        image_times = find_images('20130320_102312')
        plot_shape_progression(blob_id, image_times, mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)
if __name__ == '__main__':

    plot_one_blob()
    #main()
