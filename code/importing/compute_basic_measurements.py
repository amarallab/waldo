'''
Date:
Description:
'''

__author__ = 'peterwinter + Andrea L.'

# standard imports
import os
import sys
import math
from itertools import izip, combinations
import numpy as np

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../shared/')
PROJECT_DIR = os.path.abspath(HERE + '/../../')
EXCEPTION_DIR = PROJECT_DIR + '/data/importing/deviant/'
sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR)

# nonstandard imports
from encoding.decode_outline import decode_outline
from deviant.record_exceptions import write_pathological_input
from wio.file_manager import get_timeseries, write_timeseries_file

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
    length = 0.0
    px, py = spine[0] # previous
    for a in spine[1:]:
        cx, cy = a # current
        d = (math.sqrt((cx - px) ** 2 + (cy - py) ** 2))
        length += d
        px, py = cx, cy
    return length


def calculate_lengths_for_blob_id(blob_id, store_tmp=True, **kwargs):
    """
    Calculates and returns the timedict of lengths. Optionally inserts lengths into database.


    :param blob_id: blob identification string
    :param insert: True/False toggle to insert into database
    :param spine_entry: database document containing the list of spines.
    :return: list of lengths.
    """
    times, spines = get_timeseries(ID=blob_id, data_type='spine_rough')
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

# Ap[xy] point of the first line
# Av[xy] vector director of the first line
# Bp[xy] point of the second line
# Bv[xy] vector director of the second line
def point_and_distance_intersect_line2_line2(Apx, Apy, Avx, Avy, Bpx, Bpy, Bvx, Bvy):
    # cross product for guess the angle between lines
    d = Bvy * Avx - Bvx * Avy
    if d == 0:
        return False, 0, 0, 0 # paralels...

    # Calculate intersection
    dy = Apy - Bpy
    dx = Apx - Bpx
    ua = (Bvx * dy - Bvy * dx) / d

    px, py = Apx + ua * Avx, Apy + ua * Avy
    return True, px, py, ua   # if Av is normalized, ua is the distance and orientation!
                              # intersection is Ap + ua * Av, so if ua > 0, the intersection is "up", else is "down"


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

    # s1[xy] left middle spine point
    # s2[xy] right middle spine point
    # m[xy] middle point (between s1 and s2)
    # n[xy] vector director of the test line, perpendicular to s1-s2, and normalized
    # cross(vector) = cross(vx, vy) = (vy, -vx)

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
            # p1[xy] left point of the current outline segment
            # p2[xy] right point of the current outline segment
            lnx, lny = p2x - p1x, p2y - p1y

            # ip[xy] intersection point between perpendicular line (s1-s2) and p1-p2 line
            intersects, ipx, ipy, distance = point_and_distance_intersect_line2_line2(mx, my, nx, ny, p1x, p1y, lnx, lny)
            if intersects:
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

    # sort by distance
    less = sorted(list(less), key=lambda x: x[2])
    more = sorted(list(more), key=lambda x: x[2])

    # We can have problems with ua (is if 0?)
    if len(less) < 1:
        a = more[0]
        b = more[1]
    elif len(more) < 1:
        a = less[0]
        b = less[1]
    else:
        a = less[0]
        b = more[0]

    l = math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)  # distance between points
    return (a[0], a[1]), (b[0], b[1]), True, l

if __name__ == "__main__":
    blob_id = '20120914_172813_01708'
    blob_id = '20130320_164252_05955'
    blob_id = '20130319_150235_01501'
    compute_basic_measurements(blob_id=blob_id)
