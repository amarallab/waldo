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
