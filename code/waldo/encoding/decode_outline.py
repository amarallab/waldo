#!/usr/bin/env python
'''
Filename: decode_outline.py
Discription: contains functions involved with encoding and decoding xy coordinates between
point format (ie. tuples of (x, y) ) and an 'encoded' format that compresses
points into a series of askey characters.
'''
from __future__ import (
        absolute_import, division, print_function, unicode_literals)

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard library
import sys
import os
import math

# third party

# package imports

# globals
# for whatever reason they start counting at askey char 48
ARBIRARY_CONVERSION_FACTOR = 48

def bin1(s):
    ''' converts an integer into a string of 1s and 0s '''
    return str(s) if s <= 1 else bin(s >> 1) + str(s & 1)


def points_to_dir(pt1, pt2):
    dir_x = pt2[0] - pt1[0]
    dir_y = pt2[1] - pt1[1]

    assert type(dir_x) == int
    assert type(dir_y) == int
    assert dir_x in [-1, 0, 1]
    assert dir_y in [-1, 0, 1]

    assert dir_x == 0 or dir_y == 0
    assert dir_x != 0 or dir_y != 0

    if dir_x == 0:
        movedir = dir_y
        bit1 = '1'
    elif dir_y == 0:
        movedir = dir_x
        bit1 = '0'

    if movedir > 0:
        bit2 = '1'
    elif movedir < 0:
        bit2 = '0'

    return bit1 + bit2


def encode_outline_peter(points):
    assert type(points) in [list, tuple]
    assert (type(points[0][0]) == int) and (type(points[0][1]) == int)
    start_xy = points[0]
    steps = []
    for i, pt in enumerate(points[:-1]):
        assert type(pt) == tuple
        assert (type(pt[0]) == int) and (type(pt[1]) == int)
        steps.append(points_to_dir(points[i], points[i + 1]))

    length = len(steps)
    if len(steps) % 3 != 0: steps.append('00')
    if len(steps) % 3 != 0: steps.append('00')
    assert len(steps) % 3 == 0

    encoded_outline = ''
    for i in xrange(length / 3):
        binary = steps[(3 * i)] + steps[(3 * i) + 1] + steps[(3 * i) + 2]
        num = int(binary, base=2) + ARBIRARY_CONVERSION_FACTOR
        letter = chr(num)
        encoded_outline += letter

    x, y = start_xy
    return (x, y, length, encoded_outline)

def decode_outline_peter(outline_parts):
    # broken input if there are not three outline parts, return empty string
    if len(outline_parts) != 4:
        return []
    start_x, start_y, length, outline = outline_parts
    # check if outline parts is empty
    if start_x == '' or start_y =='':
        return []
    x, y = int(start_x), int(start_y)
    length = int(length)

    pts = []
    # go through each character in the outline string
    for o in outline:
        # convert character into integer.
        steps = ord(o)
        steps = steps - ARBIRARY_CONVERSION_FACTOR
        assert steps <= 63, 'error:(%s) is not in encoding range' % o
        assert steps >= 0, 'error:(%s) is not in encoding range' % o
        # convert intager into a binary string
        bit = str(bin1(steps))
        # remove the first two characters which
        bit = bit[2:]
        # if the binary string is less than 6 digits long,
        # add zeros to the front to make it 6 digits.
        desired_length = 6
        for i in xrange(desired_length):
            if len(bit) < desired_length:
                bit = '0' + bit
        #print(o, ord(o)-ARBIRARY_CONVERSION_FACTOR, bit, len(bit))
        def increment_loc(x, y, b):
            ''' need to use this to increment my xy coords... '''
            format_error = 'Error: boolean format is wrong:%s' % b
            unkown_error = 'Error: something unexpeted:%s' % b
            assert len(b) == 2, format_error
            assert b[0] == '0' or b[0] == '1', format_error
            assert b[1] == '0' or b[1] == '1', format_error
            if b == '00':
                x -= 1
            elif b == '01':
                x += 1
            elif b == '10':
                y -= 1
            elif b == '11':
                y += 1
            else:
                print(unkown_error)
            return x, y

        # there are three steps encoded in one character,
        # sometimes the last one or two steps are blank, hence the breaks
        if len(pts) > length: break
        pts.append((x, y))
        x, y = increment_loc(x, y, bit[:2]) # first two bits
        if len(pts) > length: break
        pts.append((x, y))
        x, y = increment_loc(x, y, bit[2:4]) # second two bits
        if len(pts) > length: break
        pts.append((x, y))
        x, y = increment_loc(x, y, bit[4:6]) # third two bits

    f = pts[0]
    l = pts[-1]
    firstlast = str(f) + str(l)

    fl_distance = math.sqrt((f[0] - l[0]) ** 2 + (f[1] - l[1]) ** 2)
    fl_error = 'first and last points not close enough to complete outline:%s' % firstlast
    len_error = 'lengths dont aggree num pts = %i, len = %i' % (len(pts), length)
    #print('distance from first to last = %f' %fl_distance)
    #assert fl_distance <= 1.5, fl_error
    #assert abs(len(pts) - length) <= 1, len_error
    return pts


def decode_outline_old(outline_parts):
    # broken input if there are not three outline parts, return empty string
    if len(outline_parts) != 3:
        return []
    start_xy, length, outline = outline_parts
    x, y = int(start_xy[0]), int(start_xy[1])
    length = int(length)
    pts = []
    # go through each character in the outline string
    for o in outline:
        # convert character into integer.
        steps = ord(o)
        steps = steps - ARBIRARY_CONVERSION_FACTOR
        assert steps <= 63, 'error:(%s) is not in encoding range' % o
        assert steps >= 0, 'error:(%s) is not in encoding range' % o
        # convert intager into a binary string
        bit = str(bin1(steps))
        # remove the first two characters which
        bit = bit[2:]
        # if the binary string is less than 6 digits long,
        # add zeros to the front to make it 6 digits.
        desired_length = 6
        for i in xrange(desired_length):
            if len(bit) < desired_length:
                bit = '0' + bit
        #print(o, ord(o)-ARBIRARY_CONVERSION_FACTOR, bit, len(bit))
        def increment_loc(x, y, b):
            ''' need to use this to increment my xy coords... '''
            format_error = 'Error: boolean format is wrong:%s' % b
            unkown_error = 'Error: something unexpeted:%s' % b
            assert len(b) == 2, format_error
            assert b[0] == '0' or b[0] == '1', format_error
            assert b[1] == '0' or b[1] == '1', format_error
            if b == '00':
                x -= 1
            elif b == '01':
                x += 1
            elif b == '10':
                y -= 1
            elif b == '11':
                y += 1
            else:
                print(unkown_error)
            return x, y

        # there are three steps encoded in one character,
        # sometimes the last one or two steps are blank, hence the breaks
        if len(pts) > length: break
        pts.append((x, y))
        x, y = increment_loc(x, y, bit[:2]) # first two bits
        if len(pts) > length: break
        pts.append((x, y))
        x, y = increment_loc(x, y, bit[2:4]) # second two bits
        if len(pts) > length: break
        pts.append((x, y))
        x, y = increment_loc(x, y, bit[4:6]) # third two bits

    f = pts[0]
    l = pts[-1]
    firstlast = str(f) + str(l)

    fl_distance = math.sqrt((f[0] - l[0]) ** 2 + (f[1] - l[1]) ** 2)
    fl_error = 'first and last points not close enough to complete outline:%s' % firstlast
    len_error = 'lengths dont aggree num pts = %i, len = %i' % (len(pts), length)
    #print('distance from first to last = %f' %fl_distance)
    #assert fl_distance <= 1.5, fl_error
    #assert abs(len(pts) - length) <= 1, len_error
    return pts

def encode_outline(points):
    assert type(points) in [list, tuple]
    assert (type(points[0][0]) == int) and (type(points[0][1]) == int)
    outline = ''
    sx, sy = points[0] # start point
    value = 0
    cur_steps = 0
    px, py = points[0] # previous points
    for p in points[1:]:
        dx = p[0] - px
        dy = p[1] - py
        px, py = p
        if dx == -1 and dy == 0:
            next_value = 0
        elif dx == 1 and dy == 0:
            next_value = 1
        elif dx == 0 and dy == -1:
            next_value = 2
        elif dx == 0 and dy == 1:
            next_value = 3
        else:
            print("E: dx=%d dy=%d" % (dx, dy))
            return (0, 0, 0, [])
        value |= next_value << (2 - cur_steps) * 2
        cur_steps += 1
        if cur_steps == 3:
            outline += chr(ARBIRARY_CONVERSION_FACTOR + value)
            value = 0
            cur_steps = 0
    if cur_steps != 0:
        outline += chr(ARBIRARY_CONVERSION_FACTOR + value)
    return (sx, sy, len(points)-1, outline)

def decode_outline(params):
    # broken input if there are not three outline parts, return empty string
    if len(params) != 4:
        return []
    start_x, start_y, length, outline = params
    # check if outline parts is empty
    if start_x == '' or start_y =='':
        return []
    x, y = int(start_x), int(start_y)
    length = int(length)

    points = [ (x, y) ]
    for ch in outline:
        byte = ord(ch) - ARBIRARY_CONVERSION_FACTOR
        assert byte <= 63, 'error:(%s) is not in encoding range' % ch
        assert byte >= 0, 'error:(%s) is not in encoding range' % ch

        for count in range(3):
            if length == 0:
                return points

            value = (byte >> 4) & 3
            if value == 0:
                x -= 1
            elif value == 1:
                x += 1
            elif value == 2:
                y -= 1
            elif value == 3:
                y += 1

            points.append( (x, y) )
            length -= 1

            byte <<= 2
    return points

def make_square():
    ''' returns a series of x,y coordinates that coorespond to the outside of a square '''
    x, y = [10], [20]

    def go_dir(x, y, num_steps=20, stepdir=(0, 0)):
        for i in xrange(num_steps):
            x.append(x[-1] + stepdir[0])
            y.append(y[-1] + stepdir[1])
        return x, y

    x, y = go_dir(x, y, stepdir=(0, 1))
    x, y = go_dir(x, y, stepdir=(1, 0))
    x, y = go_dir(x, y, stepdir=(0, -1))
    x, y = go_dir(x, y, stepdir=(-1, 0))
    return x, y


def show_worm_video(outline_timedict, window_size=20):
    from pylab import ion, plot, xlim, ylim, draw, clf
    import numpy as np

    times = sorted([float(i.replace('?', '.')) for i in outline_timedict])
    ion()
    for time, t in times[:]:
        outline = outline_timedict[t]
        # plot outline
        ox, oy = zip(*outline)
        plot(ox, oy, color='blue')
        # draw point at head.
        st_x, st_y = outline[0]
        plot([st_x], [st_y], marker='o')
        center_x = np.mean(ox)
        center_y = np.mean(oy)
        xlim([int(center_x) - window_size, int(center_x) + window_size])
        ylim([int(center_y) - window_size, int(center_y) + window_size])
        draw()
        clf()


def test_encode_decode():
    x, y = make_square()
    points = zip(x, y)
    (start_x, start_y, length, encoded_outline) = encode_outline(points)
    print('outline encoded', (start_x, start_y, length, encoded_outline))
    points2 = decode_outline((start_x, start_y, length, encoded_outline))
    for i, (pt1, pt2) in enumerate(zip(points, points2)):
        assert pt1 == pt2, 'point %i was mismatched' %i
        print(i, pt1, pt2)

if __name__ == "__main__":
    test_encode_decode()
