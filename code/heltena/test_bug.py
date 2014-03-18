import os
import sys
import euclid as eu

CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"
PROJECT_DIR = os.path.abspath(CODE_DIR)
SHARED_DIR = CODE_DIR + '/shared/'

sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

import profiling
from PIL import Image, ImageDraw
import json
import numpy as np
from importing.skeletonize_outline import compute_skeleton_from_outline as peter_compute_skeleton_from_outline
from importing.compute_basic_measurements import calculate_width_for_timepoint as peter_calculate_width_for_timepoint
import math

global minx, miny, maxx, maxy
minx = 0
miny = 0
maxx = 0
maxy = 0

def to_base(x):
    if x is None:
        return None
    return (x[0]-minx, x[1]-miny)

def heltena_calculate_width_for_timepoint_opt(spine, outline, index_along_spine=-1):
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

    middle = eu.Point2((s1[0] + s2[0]) * 0.5, (s1[1] + s2[1]) * 0.5)
    s1 = eu.Point2(*s1)
    s2 = eu.Point2(*s2)
    norm = (s2 - s1).normalize().cross()
    ls = eu.Line2(middle, norm)

    inter = set()
    outline.append(list(outline[0]))
    p1 = eu.Point2(*outline[0])
    for cur in outline[1:]:
        p2 = eu.Point2(*cur)
        if p1 != p2:
            lp = eu.Line2(p1, p2)
            intersection = ls.intersect(lp)
            if intersection is not None:
                ip = [intersection.x, intersection.y]
                if min(p1[0], p2[0]) <= ip[0] <= max(p1[0], p2[0]) and \
                   min(p1[1], p2[1]) <= ip[1] <= max(p1[1], p2[1]):
                    inter.add(tuple(ip))
            p1 = p2

    if len(inter) < 2:
        return (-1, -1), (-1, -1), False, -1

    less = []
    more = []
    for p in [eu.Point2(*p) for p in inter]:
        if ls.v[0] > ls.v[1]:
            if p.x > middle.x:
                less.append(p)
            else:
                more.append(p)
        else:
            if p.y > middle.y:
                less.append(p)
            else:
                more.append(p)

    less = sorted(list(less), key=lambda x: x.distance(middle))
    more = sorted(list(more), key=lambda x: x.distance(middle))

    if len(less) < 1:
        a = more[0]
        b = more[1]
    elif len(more) < 1:
        a = less[0]
        b = less[1]
    else:
        a = less[0]
        b = more[0]

    return (a.x, a.y), (b.x, b.y), True, (b-a).magnitude()

def point_and_distance_intersect_line2_line2(Apx, Apy, Avx, Avy, Bpx, Bpy, Bvx, Bvy):
    d = Bvy * Avx - Bvx * Avy
    if d == 0:
        return None # paralels...

    dy = Apy - Bpy
    dx = Apx - Bpx
    ua = (Bvx * dy - Bvy * dx) / d

    px, py = Apx + ua * Avx, Apy + ua * Avy
    return px, py, ua   # if Av is normalized, ua is the distance and orientation!


def heltena_calculate_width_for_timepoint(spine, outline, index_along_spine=-1):
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

def function(user):
    if user == 'heltena':
        fnc = heltena_calculate_width_for_timepoint
    else:
        fnc = peter_calculate_width_for_timepoint

    result_func = []
    error_count = 0
    for name in sys.argv[1:]:
        base = os.path.basename(name)
        #print "Using '%s'..." % base

        data = json.load(open(name, "rt"))
        spine, outline = data['pathological_input']
        spine = [np.array([np.float(i) for i in p]) for p in spine]
        outline = [np.array([np.float(i) for i in p]) for p in outline]
        #spine = [[int(i) for i in p] for p in spine]
        #outline = [[int(i) for i in p] for p in outline]

        #global minx, miny, maxx, maxy
        minx = min([x for x, y in spine + outline])
        miny = min([y for x, y in spine + outline])
        maxx = max([x for x, y in spine + outline])
        maxy = max([y for x, y in spine + outline])

        a, b, result, width = fnc(spine, outline)

        a = to_base(a)
        b = to_base(b)

        if not result:
            error_count += 1
        result_func.append("%s: %s, %s, %s, %s" % tuple([str(x) for x in ("OK" if result else "FAIL", a, b, result, width)]))

        # size = (int(maxx - minx + 1), int(maxy - miny + 1))
        #
        # spine = [tuple(to_base(v)) for v in spine]
        # outline = [tuple(to_base(v)) for v in outline]
        #
        # img = Image.new("RGB", size, "white")
        # draw = ImageDraw.Draw(img)
        # color = (128, 128, 255)
        # draw.point(spine, fill=color)
        #
        # color = (0, 0, 255)
        # draw.point(outline, fill=color)
        #
        # if result:
        #     color = (255, 0, 0)
        #     draw.point(a, fill=color)
        #     draw.point(b, fill=color)
        #
        # img.save("results/%s.png" % base, "PNG")
    return result_func

profiling.begin("Start...")
r1 = function('heltena')
profiling.end("End")
profiling.begin("Start...")
r2 = function('heltena_opt')
profiling.end("End")

if r1 != r2:
    print "NO SON IGUALES!"
else:
    print "PERFECTO!"