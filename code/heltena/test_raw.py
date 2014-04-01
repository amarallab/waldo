#!/usr/bin/env python

import math
import json
import os

def heltena_smooth_coordinate_angle(a, t, threshold):
    res = [a[0]]
    prev_p = (a[0], t[0])
    count = 0
    for p in zip(a, t):
        diff = (p[0] - prev_p[0], p[1] - prev_p[1])
        ang = math.atan2(diff[1], diff[0])
        h = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
        b = math.sin(ang) * h

        count += 1
        if abs(b) >= threshold:
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

def heltena_smooth_coordinate_plane(a, t, threshold):
    res = [a[0]]
    prev = float(a[0])
    count = 0
    for cur in a[1:]:
        count += 1
        if abs(cur-prev) >= threshold:
            d = float(cur-prev) / float(count)
            for _ in range(count):
                prev += d
                res.append(prev)
            count = 0
    if count > 0:
        d = float(cur-prev) / float(count)
        for i in range(count):
            prev += d
            res.append(prev)
    return res

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.abspath(TEST_DIR + '/../../')
TEST_DATA_DIR = PROJECT_DIR + '/code/tests/data'

save_dir = '{d}/smoothing'.format(d=TEST_DATA_DIR)
filename = '{d}/synthetic_1.json'.format(d=save_dir)

values = json.loads(open(filename, 'rt').read())
smooth_xy = values['smooth-xy']
noisy_xy = values['noisy-xy']
raw_xy = values['raw-xy']
time = values['time']
pixels_per_mm = values['pixels-per-mm']
domains = values['domains']

x = [a for a, b in raw_xy]
y = [b for a, b in raw_xy]

for threshold in [2, 5, 10, 30]:
    xx = heltena_smooth_coordinate_plane(x, time, threshold)
    yy = heltena_smooth_coordinate_plane(y, time, threshold)
    res = {'smooth-xy': zip(xx, yy),
           'noisy-xy': noisy_xy,
           'raw-xy': raw_xy,
           'time': time,
           'pixels-per-mm': pixels_per_mm,
           'domains': domains}
    json.dump(res, open('{d}/synthetic_1-plane-{t}.json'.format(d=save_dir, t=threshold), 'w'), indent=4)

    xx = heltena_smooth_coordinate_angle(x, time, threshold)
    yy = heltena_smooth_coordinate_angle(y, time, threshold)
    res = {'smooth-xy': zip(xx, yy),
           'noisy-xy': noisy_xy,
           'raw-xy': raw_xy,
           'time': time,
           'pixels-per-mm': pixels_per_mm,
           'domains': domains}
    json.dump(res, open('{d}/synthetic_1-angle-{t}.json'.format(d=save_dir, t=threshold), 'w'), indent=4)


#
#
# size = (4096, 4096)
# img = Image.new("RGB", size, "white")
# draw = ImageDraw.Draw(img)
# for filename in filenames:
#     with h5py.File(filename, "r") as f:
#         data = f['data']
#         x = data[:, 0]
#         y = data[:, 1]
#         t = f['time']
#
#         x = heltena_smooth_coordinate(x, t)
#         y = heltena_smooth_coordinate(y, t)
#
#         draw.line(zip(x, y), width=3, fill=(255, 255, 255))
#         pold = None
#         for x, y, t in zip(x, y, t):
#             if mint > t:
#                 mint = t
#             if maxt < t:
#                 maxt = t
#             p = (x, y)
#             if pold is not None:
#                 c = colorsys.hsv_to_rgb((t-mint)/(maxt-mint), 1, 1)
#                 color = (int(c[0] * 256), int(c[1] * 256), int(c[2] * 256))
#                 draw.line([pold, p], width=1, fill=color)
#             pold = p
# #out = "out/%s.png" % filename[len("dsets/"):-len(".h5")]
# out = 'out/result.png'
# img.save(out, "PNG")
# print mint, maxt