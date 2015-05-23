#!/usr/bin/env python

'''
Filename: draw_outlines_on_image.py
Description:
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from itertools import izip
#from . import manpulations #import align_outline_matricies

# nonstandard imports
#from .grab_images import create_image_directory
#from .grab_images import get_closest_image
from .grab_outlines import find_outlines_for_timepoint, \
    create_good_outline_file, grab_db_outlines, find_frame_for_time


def draw_outlines_on_single_image(image_path, outlines, savename=''):
    """
    draw outline shapes onto a single image.

    :param image_path: path for the image that should be the background
    :param list_of_outlines: list of outlines in which each outline is a list of (x,y) tuples.
    :param savename: if specified, the resulting image will be saved with this name. if not, the image will be shown.
    """
    plt.figure(1)
    ax = plt.subplot(1, 1, 1)
    label_settings = {'top': 'off', 'bottom': 'off', 'right': 'off', 'left': 'off',
                      'labelbottom': 'off', 'labeltop': 'off', 'labelright': 'off', 'labelleft': 'off'}

    background = np.asarray(Image.open(image_path)).T
    # Draw something first, should be an image.
    plt.tick_params(**label_settings)
    plt.imshow(background, cmap=cm.Greys_r)
    # plt.plot(*zip(*raw_outline), color='red')
    ax.set_aspect('equal')

    # for outline in outlines:
    #    x,y=zip(*outline)
    #    plt.plot(x, y, color='red')
    for outline_parts in outlines:
        if len(outline_parts) != 2:
            c = 'red'
            outline = outline_parts
        else:
            outline, c = outline_parts
        x, y = zip(*outline)
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


# def plot_merged_outlines(outline_list, ids):
#
#
# def before_and_after(experiment, frame1, frame2, ids1, ids2):
#     ''' creates two plots showing filled shapes for blobs
#     taken from two different frames.
#
#     params
#     -------
#     experiment: (experiment object from multiworm)
#         cooresponds to a specific ex_id
#     fame1: (int)
#         the frame number for pannel 1
#     ids1: (list of blob ids)
#         the blobs to be dispayed on pannel 1
#     fame2: (int)
#         the frame number for pannel 2
#     ids2: (list of blob ids)
#         the blobs to be dispayed on pannel 2
#     '''
#
#     def outlines_for_ids(experiment, frame, bids):
#         ''' returns lists of bids and outlines for all bids with
#         an outline found at the specified frame.
#
#         params
#         -------
#
#         experiment: (experiment object from multiworm)
#             cooresponds to a specific ex_id
#         fame: (int)
#             the frame number
#         bids: (list of blob ids)
#
#         returns
#         ------
#         a tuple containing three lists:
#         1) all blob ids for which outlines could be located
#         2) a list of all outlines (in point for ie [(x1, y1), (x2, y2) ... ])
#         3 a list of bounding boxes
#         '''
#         parser = frame_parser_spec(frame)
#         bids_w_outline, outlines, bboxes = [], [], []
#         for bid in bids:
#             blob = experiment.parse_blob(bid, parser)
#             if blob['contour_encode_len'][0]:
#                 bids_w_outline.append(bid)
#                 outline = blob_reader.decode_outline(
#                     blob['contour_start'][0],
#                     blob['contour_encode_len'][0],
#                     blob['contour_encoded'][0],
#                 )
#                 outlines.append(outline)
#                 x, y = zip(*outline)
#                 bboxes.append((min(x), min(y), max(x), max(y)))
#         return bids_w_outline, outlines, bboxes
#
#     # organize data
#     bids1, outlines1, bboxes1 = outlines_for_ids(experiment, frame1, ids1)
#     bids2, outlines2, bboxes2 = outlines_for_ids(experiment, frame2, ids2)
#     N1, N2 = len(bids1), len(bids2)
#
#     # convert to matricies
#     o1 = [outline_to_outline_matrix(o) for o in outlines1]
#     o2 = [outline_to_outline_matrix(o) for o in outlines2]
#
#     # align all the matricies to be on common coordinates.
#     aligned_matricies, bbox = align_outline_matricies(o1 + o2, bboxes1 + bboxes2)
#     o1 = aligned_matricies[:N1]
#     o2 = aligned_matricies[N1:]
#
#     print('frame1:', frame1, '| bids:', bids1)
#     print('frame2:', frame2, '| bids:', bids2)
#
#     # plots for debugging
#     # print('none of second group lost?', len(o2) == len(bids2))
#     # for o in o1 + o2:
#     #    fig, ax = plt.subplots()
#     #    ax.imshow(o)
#     #    break
#     # plt.show()
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
#     # add all entries togther into same pannel
#     if len(o1):
#         panel1 = np.zeros(o1[0].shape, dtype=int)
#         for o in o1:
#             panel1 += o
#         ax1.pcolormesh(panel1, cmap=plt.cm.YlOrBr)
#         ymax, xmax = o1[0].shape
#     if len(o2):
#         panel2 = np.zeros(o2[0].shape, dtype=int)
#         for o in o2:
#             panel2 += o
#         ymax, xmax = o2[0].shape
#         ax2.pcolormesh(panel2, cmap=plt.cm.YlOrBr)
#
#     ax2.set_xlim([0, xmax])
#     ax2.set_ylim([0, ymax])
#     ax1.set_title('frame {f}'.format(f=frame1))
#     ax2.set_title('frame {f}'.format(f=frame2))
#
#     plt.show()
#     return bbox


# def draw_outlines_on_double_image(image_path1, image_path2, outlines1, outlines2, savename=''):
#     """
#
#     :param image_path1: path for the image that should be the background for the left side of the figure
#     :param image_path2: path for the image that should be the background for the right side of the figure
#     :param outlines1: list of outlines that should be drawn on the left (each outline is a list of (x,y) tuples).
#     :param outlines2: list of outlines that should be drawn on the right (each outline is a list of (x,y) tuples).
#     :param savename: if specified, the resulting image will be saved with this name. if not, the image will be shown.
#     """
#     fig = plt.figure(1)
#
#     label_settings = {'top': 'off', 'bottom': 'off', 'right': 'off', 'left': 'off',
#                       'labelbottom': 'off', 'labeltop': 'off', 'labelright': 'off', 'labelleft': 'off'}
#
#     fig.set_size_inches(18.5, 13.5)
#     ax = plt.subplot(1, 2, 1)
#
#     background1 = np.asarray(Image.open(image_path1)).T
#     plt.tick_params(**label_settings)
#     plt.imshow(background1, cmap=cm.Greys_r)
#     ax.set_aspect('equal')
#     plt.tick_params(**label_settings)
#     ax.set_xlim([0, len(background1[0])])
#     ax.set_ylim([0, len(background1)])
#     for outline_parts in outlines1:
#         if len(outline_parts) != 2:
#             c = 'red'
#             outline = outline_parts
#         else:
#             outline, c = outline_parts
#         x, y = zip(*outline)
#         plt.plot(x, y, color=c)
#
#     ax = plt.subplot(1, 2, 2)
#     background2 = np.asarray(Image.open(image_path2)).T
#     plt.tick_params(**label_settings)
#     plt.imshow(background2, cmap=cm.Greys_r)
#     ax.set_aspect('equal')
#     plt.tick_params(**label_settings)
#     ax.set_xlim([0, len(background2[0])])
#     ax.set_ylim([0, len(background2)])
#     for outline_parts in outlines2:
#         if len(outline_parts) != 2:
#             c = 'red'
#             outline = outline_parts
#         else:
#             outline, c = outline_parts
#         x, y = zip(*outline)
#         plt.plot(x, y, color=c)
#
#     if savename:
#         plt.savefig(savename)
#     else:
#         plt.show()
#     plt.clf()


def color_outlines(outline_lists, colors=['red', 'yellow', 'blue']):
    colored_outlines = []
    colors = colors[:len(outline_lists)]
    for outline_list, color in zip(outline_lists, colors):
        for outline in outline_list:
            colored_outlines.append((outline, color))
    return colored_outlines


# def get_image_and_outlines(ex_id, pictime):
#     """
#     Return a numpy array of an image and a list of points for shape outlines for an experiment ID.
#     While it may not return the image and outlines from the exact time, it returns them for the closest
#     time in which an image was found.
#
#     :param ex_id: experiment ID
#     :param pictime: the time in seconds that we wish to get close to.
#     :return: (numpy array of image, list of shape outlines each list is a list of x, y positions)
#     """
#     all_outlines, good_outlines, db_outlines = [], [], []
#     print
#     'time chosen:', pictime
#     image_times = create_image_directory(ex_id=ex_id)
#     closest_image_time, closest_image = get_closest_image(target_time=pictime, image_dict=image_times)
#     print
#     'closest image time:', closest_image_time
#     # print image_times
#     frame, timepoint = find_frame_for_time(ex_id=ex_id, time=closest_image_time)
#     print
#     'closest frame and time:', frame, timepoint
#     all_outlines = find_outlines_for_timepoint(ex_id=ex_id, frame=frame)
#     good_outlines = create_good_outline_file(ex_id=ex_id, frame=frame)
#     db_outlines = grab_db_outlines(ex_id=ex_id, timepoint=timepoint)
#     print
#     'all_outlines', len(all_outlines)
#     print
#     'size_outlines', len(good_outlines)
#     print
#     'db_outlines', len(db_outlines)
#
#     return closest_image, all_outlines, good_outlines, db_outlines


def single_image(experiment, pictime=500):
    print
    'time chosen:', pictime
    #image_times = create_image_directory(ex_id=ex_id)
    image_times = experiment.image_files
    closest_image_time, closest_image = get_closest_image(target_time=pictime, image_dict=image_times)
    print
    'closest image time:', closest_image_time
    # print image_times
    frame, timepoint = find_frame_for_time(ex_id=ex_id, time=closest_image_time)
    print
    'closest frame and time:', frame, timepoint
    all_outlines = find_outlines_for_timepoint(ex_id=ex_id, frame=frame)
    db_outlines = grab_db_outlines(ex_id=ex_id, timepoint=timepoint)
    outlines = color_outlines(outline_lists=[all_outlines, db_outlines], colors=['red', 'blue'])
    draw_outlines_on_single_image(closest_image, outlines)
