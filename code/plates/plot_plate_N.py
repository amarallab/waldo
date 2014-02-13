#!/usr/bin/env python

'''
Filename: plot_plate_N.py
Description: creates figures from data in database.
'''
__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
from itertools import izip
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import Image
import matplotlib.cm as cm
import math

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_directory)

# nonstandard imports
from Shared.Code.ImageManipulation.draw_outlines_on_image import get_image_and_outlines, color_outlines
import Shared.Code.Database.mongo_support_functions as mongo
from Shared.Code.Database.mongo_retrieve import mongo_query
from Shared.Code.Settings.data_settings import mongo_settings
from compute_N_for_plate import compute_N

def plot_N_timeseries_and_images(ex_id, **kwargs):

    savename = './../Results/N-Plots/{ex_id}.png'.format(ex_id=ex_id)
    stage1_data = mongo_query({'ex_id': ex_id, 'type': 'stage1'}, find_one=True, col='plate_collection', **kwargs)['data']
    times, all_N, good_N = compute_N(ex_id)

    # grab images and outlines
    im1, all_outlines1, good_outlines1, db_outlines1 = get_image_and_outlines(ex_id=ex_id, pictime=500)
    im2, all_outlines2, good_outlines2, db_outlines2 = get_image_and_outlines(ex_id=ex_id, pictime=1000)
    outlines1 = color_outlines([all_outlines1, good_outlines1, db_outlines1])
    outlines2 = color_outlines([all_outlines2, good_outlines2, db_outlines2])

    print len(all_outlines1), len(all_outlines2)
    print len(good_outlines1), len(good_outlines2)

    # definitions for the axes
    left, mid, right = 0.1, 0.5, 0.9
    margin = 0.1
    left_img_mid, right_img_mid = mid - 0.5 * margin, mid + 0.5 * margin
    top, split, bottom = .9, 0.7, .1
    img_top, timeseries_bod = split - 0.5 *margin, split + 0.5 * margin
    t_width, t_height = right - left, top - timeseries_bod

    img_width, img_height = math.fabs(left - left_img_mid), split - bottom - 0.5 *margin

    # start with a rectangular Figure
    fig = plt.figure(1)
    fig.set_size_inches(25, 15)

    # define all plots and set locations
    time_rect = [left, timeseries_bod, t_width, t_height]
    N_timeseries = plt.axes(time_rect)

    left_img_rect = [left, bottom, img_width, img_height]
    right_img_Rect = [right_img_mid, bottom, img_width, img_height]
    print left_img_rect
    print right_img_Rect
    img1 = plt.axes(left_img_rect)
    img2 = plt.axes(right_img_Rect)
    label_settings = {'top': 'off', 'bottom': 'off', 'right': 'off', 'left': 'off',
                      'labelbottom': 'off', 'labeltop': 'off', 'labelright': 'off', 'labelleft': 'off'}

    N_timeseries.fill_between(times, 0, all_N, color='red', label='All')
    N_timeseries.fill_between(times, 0, good_N, color='yellow', label='Worm-Sized')
    N_timeseries.fill_between(stage1_data['time'], 0, stage1_data['N'], color='blue', label='Analyzed')
    N_timeseries.set_xlim([1, 1200])
    #N_timeseries.plot(times, all_N, color='red', label='All')
    #N_timeseries.plot(times, good_N, color='yellow', label='Worm-Sized')
    #N_timeseries.plot(stage1_data['time'], stage1_data['N'], color='blue', label='Analyzed')
    N_timeseries.legend()

    background1 = np.asarray(Image.open(im1)).T
    img1.imshow(background1, cmap=cm.Greys_r)
    img1.tick_params(**label_settings)

    img1.set_aspect('equal')
    img1.set_xlim([0, len(background1[0])])
    img1.set_ylim([0, len(background1)])

    for outline_parts in outlines1:
        if len(outline_parts) != 2:
            c = 'red'
            outline = outline_parts
        else:
            outline, c = outline_parts
        x,y=zip(*outline)
        img1.plot(x, y, color=c)

    background2 = np.asarray(Image.open(im2)).T
    img2.imshow(background2, cmap=cm.Greys_r)
    img2.tick_params(**label_settings)

    img2.set_aspect('equal')
    img2.set_xlim([0, len(background2[0])])
    img2.set_ylim([0, len(background2)])
    for outline_parts in outlines2:
        if len(outline_parts) != 2:
            c = 'red'
            outline = outline_parts
        else:
            outline, c = outline_parts
        x,y=zip(*outline)
        img2.plot(x, y, color=c)

    print os.path.abspath(savename)

    #plt.show()
    plt.savefig(savename)
    plt.clf()

def main(ex_ids):
    """
    runs plot_speed_of_plate() repeatidly for a list of ex_ids and maintaining the same mongo_connection.
    """
    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['plate_collection'])
    try:
        for ex_id in ex_ids:
            #plot_N_timeseries_and_images(ex_id, mongo_client=mongo_client)
            try:
                print ex_id
                plot_N_timeseries_and_images(ex_id, mongo_client=mongo_client)

            except Exception as e:
                print e
                print 'WARNING:', ex_id, 'not plotted'

    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)


if __name__ == '__main__':
    ex_ids = ['20130323_170511', '20130319_134745', '20130320_102312', '20130409_172434', '20130422_115041',
              '20130426_115024']
    ex_ids = ['20131108_153706', '20131108_160633', '20131108_164129']
    main(ex_ids=ex_ids[:])
