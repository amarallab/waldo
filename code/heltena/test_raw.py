#!/usr/bin/env python

import h5py
from PIL import Image, ImageDraw
import sys
from glob import glob
import colorsys
import math

filenames = []
for arg in sys.argv[1:]:
    filenames += glob(arg)

mint = 0
maxt = 3600
THRESHOLD = 30

def optimize(a, t):

    res =[a[0]]
    prev_p = (a[0], t[0])
    count = 0
    for p in zip(a, t):
        diff = (p[0] - prev_p[0], p[1] - prev_p[1])
        ang = math.atan2(diff[1], diff[0])
        h = math.sqrt(diff[0] ** 2  + diff[1] ** 2)
        b = math.sin(ang) * h

        count += 1
        if abs(b) >= THRESHOLD:
            diff = (diff[0] / float(count), diff[1] / float(count))
            for _ in range(count):
                prev_p = (prev_p[0] + diff[0], prev_p[1] + diff[1])
                res.append(prev_p[0])
            count = 0

    if count > 0:
        diff = (p[0] - prev_p[0], p[1] - prev_p[1])
        diff = (diff[0] / float(count), diff[1] / float(count))
        for _ in range(count):
            prev_p = (prev_p[0] + diff[0], prev_p[1] + diff[1])
            res.append(prev_p[0])

    return res

    # res = [a[0]]
    # prev = float(a[0])
    # count = 0
    # for cur in a[1:]:
    #     count += 1
    #     if abs(cur-prev) >= THRESHOLD:
    #         d = float(cur-prev) / float(count)
    #         for _ in range(count):
    #             prev += d
    #             res.append(prev)
    #         count = 0
    # if count > 0:
    #     d = float(cur-prev) / float(count)
    #     for i in range(count):
    #         prev += d
    #         res.append(prev)
    # return res

size = (4096, 4096)
img = Image.new("RGB", size, "white")
draw = ImageDraw.Draw(img)
for filename in filenames:
    with h5py.File(filename, "r") as f:
        data = f['data']
        x = data[:, 0]
        y = data[:, 1]
        t = f['time']

        x = optimize(x, t)
        y = optimize(y, t)

        draw.line(zip(x, y), width=3, fill=(255, 255, 255))
        pold = None
        for x, y, t in zip(x, y, t):
            if mint > t:
                mint = t
            if maxt < t:
                maxt = t
            p = (x, y)
            if pold is not None:
                c = colorsys.hsv_to_rgb((t-mint)/(maxt-mint), 1, 1)
                color = (int(c[0] * 256), int(c[1] * 256), int(c[2] * 256))
                draw.line([pold, p], width=1, fill=color)
            pold = p
#out = "out/%s.png" % filename[len("dsets/"):-len(".h5")]
out = 'out/result.png'
img.save(out, "PNG")
print mint, maxt