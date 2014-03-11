#!/usr/bin/env python

from PIL import Image, ImageDraw
from data import outline, result

outline = [(1000, 1000), (1000, 1001), (1000, 1002), (1000, 1003), (1001, 1003), (1001, 1004), (1001, 1005), (1001, 1006), (1002, 1006), (1003, 1006), (1003, 1005), (1004, 1005), (1004, 1004), (1004, 1003), (1003, 1003), (1003, 1002), (1004, 1002), (1004, 1001), (1004, 1000), (1004, 999), (1005, 999), (1005, 998), (1005, 997), (1006, 997), (1006, 996), (1007, 996), (1007, 995), (1008, 995), (1009, 995), (1010, 995), (1011, 995), (1011, 994), (1012, 994), (1013, 994), (1014, 994), (1015, 994), (1016, 994), (1016, 993), (1017, 993), (1018, 993), (1019, 993), (1019, 992), (1020, 992), (1021, 992), (1021, 991), (1021, 990), (1022, 990), (1022, 989), (1022, 988), (1022, 987), (1021, 987), (1021, 986), (1020, 986), (1020, 987), (1019, 987), (1018, 987), (1018, 988), (1017, 988), (1017, 989), (1016, 989), (1016, 990), (1015, 990), (1014, 990), (1013, 990), (1013, 991), (1012, 991), (1011, 991), (1010, 991), (1009, 991), (1008, 991), (1007, 991), (1007, 992), (1006, 992), (1005, 992), (1004, 992), (1004, 993), (1003, 993), (1003, 994), (1002, 994), (1002, 995), (1002, 996), (1001, 996), (1001, 997), (1001, 998), (1001, 999)]

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
