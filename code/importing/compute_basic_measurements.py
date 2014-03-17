'''
Date:
Description:
'''

__author__ = 'peterwinter + Andrea L.'

# standard imports
import os
import sys
import math
import time
from itertools import izip, combinations
from pylab import *
import numpy as np

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../shared/')
PROJECT_DIR = os.path.abspath(HERE + '/../../')
EXCEPTION_DIR = PROJECT_DIR + '/data/importing/exceptions/'
sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR)

# nonstandard imports
from Encoding.decode_outline import pull_smoothed_outline, decode_outline
from GeometricCalculations import get_ortogonal_to_spine, find_intersection_points, check_point_is_inside_box, calculate_area_of_box
from ExceptionHandling.record_exceptions import write_pathological_input
from shared.wio.file_manager import get_timeseries, write_timeseries_file

def compute_basic_measurements(blob_id, verbose=True, **kwargs):

    lengths = calculate_lengths_for_blob_id(blob_id, **kwargs)
    if verbose:
        try:
            m = round(np.mean(lengths), ndigits=2)
            s = round(np.std(lengths), ndigits=2)
            print '\tlengths mean:{m} | std:{s} | N:{N}'.format(m=m, s=s, N=len(lengths))
        except:
            print '\tcould not compute mean and std for lengths'
    w20, w50, w80 = calculate_widths_for_blob_id(blob_id, **kwargs)
    if verbose:
        print '\twidths at 20% | 50% | 80% along spine'
        print '\twidth N: {N20} | {N50} | {N80}'.format(N20=len(w20), N50=len(w50), N80=len(w80))
        try:
            ms = [round(np.mean(w), ndigits=2) for w in [w20, w50, w80]]
            ss = [round(np.std(w), ndigits=2) for w in [w20, w50, w80]]
            ms = ' | '.join(map(str, ms))
            ss = ' | '.join(map(str, ss))
            print '\twidth means: {ms}'.format(ms=ms)
            print '\twidth stds: {ss}'.format(ss=ss)
        except:
            print '\tcould not compute mean and std for widths'

def calculate_length_for_timepoint(spine):
    # length = 0.0
    # for i, a in enumerate(spine[:-1]):
    #     x1, y1 = spine[i]
    #     x2, y2 = spine[i + 1]
    #     d = (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    #     length += d
    # return length
    length = 0.0
    px, py = spine[0] # previous
    for a in spine[1:]:
        cx, cy = a # current
        d = (math.sqrt((cx - px) ** 2 + (cy - py) ** 2))
        length += d
        px, py = cx, cy
    return length


def calculate_lengths_for_blob_id(blob_id, times=[], store_tmp=True, **kwargs):
    """
    Calculates and returns the timedict of lengths. Optionally inserts lengths into database.


    :param blob_id: blob identification string
    :param insert: True/False toggle to insert into database
    :param spine_entry: database document containing the list of spines.
    :return: timedict of lengths.
    """
    times, spines = get_timeseries(ID=blob_id, data_type='spine_rough', **kwargs)
    lengths = []
    for spine in spines:
        l = 0
        if len(spine) > 0 and not np.isnan(spine[0][0]):
            l = calculate_length_for_timepoint(spine)
        lengths.append(l)
    data_type = 'length_rough'
    if store_tmp:
        write_timeseries_file(ID=blob_id, data_type=data_type,
                              times=times, data=lengths)
    return lengths


def calculate_widths_for_blob_id(blob_id, store_tmp=True, **kwargs):
    """
    calculates and returns three timedicts with the widths at 20, 50, and 80% of the length of the worm.

    :param blob_id: blob identification string
    :param insert: True/False toggle to insert into database
    :param spine_entry: database document containing the list of spines.
    :return: a tuple of three width timedicts.
    """
    # if temp data is cached, use that instead of querying database
    times, spines = get_timeseries(ID=blob_id, data_type='spine_rough')
    times, encoded_outlines = get_timeseries(ID=blob_id, data_type='encoded_outline')    
    width20, width50, width80 = [], [], []
    for spine, en_outline in izip(spines, encoded_outlines):
        # make sure to catch not a number. not sure if necessary.
        if len(spine) > 0 and np.isnan(spine[0][0]):
            spine = []
        outline = decode_outline(en_outline)
        if len(spine) == 50:
            p1, p2, flag1, w20 = calculate_width_for_timepoint(spine, outline, index_along_spine=10)
            p1, p2, flag2, w50 = calculate_width_for_timepoint(spine, outline, index_along_spine=25)
            p1, p2, flag3, w80 = calculate_width_for_timepoint(spine, outline, index_along_spine=40)
        else:
            w20 = w50 = w80 = 0
                                                               
        width20.append(w20)
        width50.append(w50)
        width80.append(w80)

    if store_tmp:
        write_timeseries_file(ID=blob_id, data_type='width20',
                              times=times, data=width20)
        write_timeseries_file(ID=blob_id, data_type='width50',
                              times=times, data=width50)
        write_timeseries_file(ID=blob_id, data_type='width80',
                              times=times, data=width80)

    return width20, width50, width80

'''
def show_worm_video(spine_timedict, outline_timedict):
    # floats sort more properly than strings, hence using sorted tuple with (float, string)
    times = sorted([(float(t.replace('?', '.')), t) for t in spine_timedict])

    ion()
    for t_float, t in times[:]:
        p1, p2, flag, width = calculate_width_for_timepoint(spine_timedict[t], outline_timedict[t])

        spine = spine_timedict[t]
        outline = outline_timedict[t]

        #print t
        sx = [v[0] for v in spine]
        sy = [v[1] for v in spine]
        ox = [v[0] for v in outline]
        oy = [v[1] for v in outline]

        if flag == True:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='o', color='green')
            plot(sx, sy, color='blue')
            plot(ox, oy, color='blue')

        else:
            plot(sx, sy, color='red')
            plot(ox, oy, color='red')
            #import pickle
            #pickle.dump((spine, outline), open('spine_outline.pkl', 'w'))
            #exit()

        center_x, center_y = spine[len(spine) / 2]
        #plot([xys[0][0]], [xys[0][1]], marker='o', color='red')
        #plot([xys[1][0]], [xys[1][1]], marker='o', color='blue')
        window_size = 30
        xlim([int(center_x) - window_size, int(center_x) + window_size])
        ylim([int(center_y) - window_size, int(center_y) + window_size])
        draw()

        if flag == False:
            time.sleep(1)

        clf()
'''

def point_and_distance_intersect_line2_line2(Apx, Apy, Avx, Avy, Bpx, Bpy, Bvx, Bvy):
    d = Bvy * Avx - Bvx * Avy
    if d == 0:
        return None # paralels...

    dy = Apy - Bpy
    dx = Apx - Bpx
    ua = (Bvx * dy - Bvy * dx) / d

    px, py = Apx + ua * Avx, Apy + ua * Avy
    return px, py, ua   # if Av is normalized, ua is the distance and orientation!


def calculate_width_for_timepoint(spine, outline, index_along_spine=-1):
    if index_along_spine == -1:
        index_along_spine = len(spine) / 2

    s1 = spine[index_along_spine]
    if index_along_spine + 1 != len(spine):
        s2 = spine[index_along_spine+1]
    else:
        s2 = s1
        s1 = spine[index_along_spine-1]

    if np.array_equal(s1, s2):
        s2[0] += 1e-6

    s1x, s1y = s1[0], s1[1]
    s2x, s2y = s2[0], s2[1]
    mx, my = (s1x + s2x) * 0.5, (s1y + s2y) * 0.5
    nx, ny = s2x - s1x, s2y - s1y
    l = math.sqrt(nx**2 + ny**2)
    # normalize and cross
    nx, ny = ny / l, - nx / l

    inter = set()
    outline.append(list(outline[0]))
    p1x, p1y = outline[0][0], outline[0][1]
    for cur in outline[1:]:
        p2x, p2y = cur[0], cur[1]
        if p1x != p2x and p1y != p2y:
            lnx, lny = p2x - p1x, p2y - p1y
            ipx, ipy, distance = point_and_distance_intersect_line2_line2(mx, my, nx, ny, p1x, p1y, lnx, lny)
            if distance is not None:
                minx, maxx = (p1x, p2x) if p1x < p2x else (p2x, p1x)
                miny, maxy = (p1y, p2y) if p1y < p2y else (p2y, p1y)
                if ipx+0.5 >= minx and ipx-0.5 <= maxx and \
                   ipy+0.5 >= miny and ipy-0.5 <= maxy:
                    inter.add((ipx, ipy, distance))
            p1x, p1y = p2x, p2y

    if len(inter) < 2:
        return (-1, -1), (-1, -1), False, -1

    less = []
    more = []
    for x, y, d in inter:
        if d < 0:
            less.append((x, y, -d))
        else:
            more.append((x, y, d))

    less = sorted(list(less), key=lambda x: x[2])
    more = sorted(list(more), key=lambda x: x[2])

    if len(less) < 1:
        a = more[0]
        b = more[1]
    elif len(more) < 1:
        a = less[0]
        b = less[1]
    else:
        a = less[0]
        b = more[0]

    l = math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return (a[0], a[1]), (b[0], b[1]), True, l


# def calculate_width_for_timepoint(spine, outline, index_along_spine=-1):
#     '''
#         spine and ouline are list of [x,y]
#         index_along_spine is len(spine)/2 for the midpoint
#         returns:
#
#         (x1,y1) and (x2,y2) of the intersection points
#         True if everything is ok
#         width
#
#         if flag is False, function returns
#         (-1,-1), (-1,-1), False, -1
#     '''
#
#     if index_along_spine == -1:
#         index_along_spine = len(spine) / 2
#
#     outline.append(list(outline[0]))
#     orthogonal_m, q1 = get_ortogonal_to_spine(spine, index_along_spine)
#
#     intersection_xs, intersection_ys = find_intersection_points(q1, orthogonal_m, outline)
#
#     if len(intersection_xs) == 0:
#         #print 'intersection points are zero!'
#         write_pathological_input((spine, outline), input_type='spine/outline',
#                                  note='no intersection points',
#                                  savename='%sno_intersection_%s.json' % (EXCEPTION_DIR, str(time.time())))
#         return (-1, -1), (-1, -1), False, -1
#
#     # if len(intersection_xs) % 2 != 0:
#     #     #print 'intersection points are odd', len(intersection_xs)
#     #     write_pathological_input((spine, outline), input_type='spine/outline',
#     #                              note='num intersection points odd',
#     #                              savename='%sodd_num_intersection_%s.json' % (EXCEPTION_DIR, str(time.time())))
#     #     #print intersection_xs, intersection_ys
#     #     return (intersection_xs[0], intersection_ys[0]), (-1, -1), False, -1
#
#
#     #HELTENA changes... I think now it's look better
#     # for i, a in enumerate(points):
#     #     for j, b in enumerate(points[i + 1:], start=i + 1):
#     #         condition, area = check_point_is_inside_box(q1, points[i], points[j]) # function changed!!!
#     #         if condition and area < min_area:
#     #             min_area = area
#     #             point_pair = [points[i], points[j]]
#
#     #points = sorted(zip(intersection_xs, intersection_ys)) # Why are you sorting?
#     points = zip(intersection_xs, intersection_ys)
#     min_area = 1e200
#     point_pair = []
#     for a, b in combinations(points, 2):
#         area = calculate_area_of_box(a, b)
#         if area < min_area:
#             if check_point_is_inside_box(q1, a, b):
#                 min_area = area
#                 point_pair = [a, b]
#
#     if len(point_pair) == 0:
#         #print 'spine outside of outline'
#         write_pathological_input((spine, outline), input_type='spine/outline', note='spine outside outline',
#                                  savename='%sspine_outside_outline_%s.json' % (EXCEPTION_DIR, str(time.time())))
#         return -1, -1, False, -1
#     else:
#         # width = math.sqrt((point_pair[0][0] - point_pair[1][0]) ** 2 + (point_pair[0][1] - point_pair[1][1]) ** 2)
#         # return point_pair[0], point_pair[1], True, width
#         a = point_pair[0]
#         b = point_pair[1]
#         width = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
#         return a, b, True, width


if __name__ == "__main__":
    blob_id = '20120914_172813_01708'
    blob_id = '20130320_164252_05955'
    blob_id = '20130319_150235_01501'
    compute_basic_measurements(blob_id=blob_id)
