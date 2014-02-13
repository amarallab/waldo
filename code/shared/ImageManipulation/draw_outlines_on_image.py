#!/usr/bin/env python

'''
Filename: draw_outlines_on_image.py
Description:
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import izip

# nonstandard imports
from grab_images import create_image_directory, get_closest_image
from grab_outlines import find_outlines_for_timepoint, create_good_outline_file, grab_db_outlines, find_frame_for_time

def draw_outlines_on_single_image(image_path, outlines, savename=''):
    """
    draw outline shapes onto a single image.

    :param image_path: path for the image that should be the background
    :param list_of_outlines: list of outlines in which each outline is a list of (x,y) tuples.
    :param savename: if specified, the resulting image will be saved with this name. if not, the image will be shown.
    """
    plt.figure(1)
    ax = plt.subplot(1,1,1)
    label_settings = {'top': 'off', 'bottom': 'off', 'right': 'off', 'left': 'off',
                      'labelbottom': 'off', 'labeltop': 'off', 'labelright': 'off', 'labelleft': 'off'}

    background = np.asarray(Image.open(image_path)).T
    # Draw something first, should be an image.
    plt.tick_params(**label_settings)
    plt.imshow(background, cmap=cm.Greys_r)
    #plt.plot(*zip(*raw_outline), color='red')
    ax.set_aspect('equal')

    #for outline in outlines:
    #    x,y=zip(*outline)
    #    plt.plot(x, y, color='red')
    for outline_parts in outlines:
        if len(outline_parts) != 2:
            c = 'red'
            outline = outline_parts
        else:
            outline, c = outline_parts
        x,y=zip(*outline)
        plt.plot(x, y, color=c)
    plt.tick_params(**label_settings)
    ax.set_aspect('equal')
    ax.set_xlim([0, len(background[0])])
    ax.set_ylim([0, len(background)])
    ax.invert_yaxis()
    if savename:
        plt.savefig(savename)
    else:
        plt.show()
    plt.clf()


def draw_outlines_on_double_image(image_path1, image_path2, outlines1, outlines2, savename=''):

    """

    :param image_path1: path for the image that should be the background for the left side of the figure
    :param image_path2: path for the image that should be the background for the right side of the figure
    :param outlines1: list of outlines that should be drawn on the left (each outline is a list of (x,y) tuples).
    :param outlines2: list of outlines that should be drawn on the right (each outline is a list of (x,y) tuples).
    :param savename: if specified, the resulting image will be saved with this name. if not, the image will be shown.
    """
    fig = plt.figure(1)

    label_settings = {'top': 'off', 'bottom': 'off', 'right': 'off', 'left': 'off',
                      'labelbottom': 'off', 'labeltop': 'off', 'labelright': 'off', 'labelleft': 'off'}

    fig.set_size_inches(18.5, 13.5)
    ax = plt.subplot(1,2,1)

    background1 = np.asarray(Image.open(image_path1)).T
    plt.tick_params(**label_settings)
    plt.imshow(background1, cmap=cm.Greys_r)
    ax.set_aspect('equal')
    plt.tick_params(**label_settings)
    ax.set_xlim([0, len(background1[0])])
    ax.set_ylim([0, len(background1)])
    for outline_parts in outlines1:
        if len(outline_parts) != 2:
            c = 'red'
            outline = outline_parts
        else:
            outline, c = outline_parts
        x,y=zip(*outline)
        plt.plot(x, y, color=c)

    ax = plt.subplot(1,2,2)
    background2 = np.asarray(Image.open(image_path2)).T
    plt.tick_params(**label_settings)
    plt.imshow(background2, cmap=cm.Greys_r)
    ax.set_aspect('equal')
    plt.tick_params(**label_settings)
    ax.set_xlim([0, len(background2[0])])
    ax.set_ylim([0, len(background2)])
    for outline_parts in outlines2:
        if len(outline_parts) != 2:
            c = 'red'
            outline = outline_parts
        else:
            outline, c = outline_parts
        x,y=zip(*outline)
        plt.plot(x, y, color=c)

    if savename:
        plt.savefig(savename)
    else:
        plt.show()
    plt.clf()

def color_outlines(outline_lists, colors=['red', 'yellow', 'blue']):
    colored_outlines = []
    colors = colors[:len(outline_lists)]
    for outline_list, color in izip(outline_lists, colors):
        for outline in outline_list:
            colored_outlines.append((outline, color))
    return colored_outlines


def get_image_and_outlines(ex_id, pictime):
    """
    Return a numpy array of an image and a list of points for shape outlines for an experiment ID.
    While it may not return the image and outlines from the exact time, it returns them for the closest
    time in which an image was found.

    :param ex_id: experiment ID
    :param pictime: the time in seconds that we wish to get close to.
    :return: (numpy array of image, list of shape outlines each list is a list of x, y positions)
    """
    all_outlines, good_outlines, db_outlines = [], [], []
    print 'time chosen:', pictime
    image_times = create_image_directory(ex_id=ex_id)
    closest_image_time, closest_image = get_closest_image(target_time=pictime, image_dict=image_times)
    print 'closest image time:', closest_image_time
    #print image_times
    frame, timepoint = find_frame_for_time(ex_id=ex_id, time=closest_image_time)
    print 'closest frame and time:', frame, timepoint
    all_outlines = find_outlines_for_timepoint(ex_id=ex_id, frame=frame)
    good_outlines = create_good_outline_file(ex_id=ex_id, frame=frame)
    db_outlines = grab_db_outlines(ex_id=ex_id, timepoint=timepoint)
    print 'all_outlines', len(all_outlines)
    print 'size_outlines', len(good_outlines)
    print 'db_outlines', len(db_outlines)

    return closest_image, all_outlines, good_outlines, db_outlines


def single_image(ex_id, pictime=500):
    print 'time chosen:', pictime
    image_times = create_image_directory(ex_id=ex_id)
    closest_image_time, closest_image = get_closest_image(target_time=pictime, image_dict=image_times)
    print 'closest image time:', closest_image_time
    #print image_times
    frame, timepoint = find_frame_for_time(ex_id=ex_id, time=closest_image_time)
    print 'closest frame and time:', frame, timepoint
    all_outlines = find_outlines_for_timepoint(ex_id=ex_id, frame=frame)
    db_outlines = grab_db_outlines(ex_id=ex_id, timepoint=timepoint)
    outlines = color_outlines(outline_lists=[all_outlines, db_outlines], colors=['red', 'blue'])
    draw_outlines_on_single_image(closest_image, outlines)

if __name__ == '__main__':

    # toggles
    ex_id = '20130323_170511'
    #ex_id = '20130319_134745'
    #ex_id = '20130320_102312'
    #ex_id = '20130409_172434'
    ex_id = '20130422_115041'
    #ex_id = '20130426_115024'
    '''
    im1, all_outlines1, good_outlines1, db_outlines1 = get_image_and_outlines(ex_id=ex_id, pictime=500)
    im2, all_outlines2, good_outlines2, db_outlines2 = get_image_and_outlines(ex_id=ex_id, pictime=3500)
    outlines1 = color_outlines([all_outlines1, good_outlines1, db_outlines1])
    outlines2 = color_outlines([all_outlines2, good_outlines2, db_outlines2])
    draw_outlines_on_double_image(image_path1=im1, image_path2=im2, outlines1=outlines1, outlines2=outlines2)
    '''
    single_image(ex_id=ex_id)
