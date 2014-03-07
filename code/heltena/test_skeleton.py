#!/usr/bin/env python

from PIL import Image, ImageDraw
from data import outline, result

def bbox(a):
    min = list(a[0])
    max = list(a[0])
    for p in a[1:]:
        for i in range(2):
            if min[i] > p[i]:
                min[i] = p[i]
            if max[i] < p[i]:
                max[i] = p[i]
    return tuple(min), tuple(max)
 
min, max = bbox(outline)
size = tuple([a-i for i, a in zip(min,max)])
points = [(x-min[0], y-min[1]) for x, y in outline]
result = [(x-min[0], y-min[1]) for x, y in result]

print "SIZE: %s, min: %s, max: %s" % (size, min, max)
img = Image.new("RGB", size, "white")
draw = ImageDraw.Draw(img)
color = (255, 0, 0)
draw.polygon(points, fill=color)
color = (0, 0, 255)
draw.polygon(result, fill=color)
img.save("result.png", "PNG")
