from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

def calculate_fingerprint(image_filename, mask_filename):
    _image = Image.open(image_filename)
    _mask = Image.open(mask_filename)
    image = np.array(_image.getdata())
    mask = np.array(_mask.getdata())
    image *= mask > 0
    image = image.astype('float') / 256.0
    width = _image.size[0]
    height = _image.size[1]

    def move_pixel(x, y):
        x += 1
        if x >= width:
            x = 0
            y += 1
        return x, y

#     ix, iy = 0, 0
#     for v in image:
#         print "X" if v > 0 else " ",
#         ix, iy = move_pixel(ix, iy)
#         if ix == 0:
#             print ""

    plus_scores = []
    minus_scores = []
    dist_scores = []
    ix, iy = 0, 0
    for i in range(len(image)-1):
        iv = image[i]
        if iv > 0:
            jx, jy = move_pixel(ix, iy)
            for j in range(i+1, len(image)):
                jv = image[j]
                if jv > 0:
                    plus_ij = iv * iv + jv * jv
                    minus_ij = abs(iv * iv - jv * jv)
                    dist = math.hypot(jx-ix, jy-iy)
                
                    plus_scores.append(plus_ij)
                    minus_scores.append(minus_ij)
                    dist_scores.append(dist)
                
                jx, jy = move_pixel(jx, jy)
        ix, iy = move_pixel(ix, iy)

    plus_heatmap, xedges, yedges = np.histogram2d(dist_scores, plus_scores, bins=50)
    plus_extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    minus_heatmap, xedges, yedges = np.histogram2d(dist_scores, minus_scores, bins=50)
    minus_extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    f, ax = plt.subplots(1,2)
    f.set_size_inches((6,3))
    ax[0].imshow(plus_heatmap, extent=plus_extent, aspect='auto')
    ax[1].imshow(minus_heatmap, extent=minus_extent, aspect='auto')


def calculate_from_id(id):
    BASEDIR = "/Users/heltena/src/waldo/data/cutouts/20130702_135704"
    TEST_BASE_NAME = "1167_0/{id}".format(id=id)
    TEST_IMAGE_FILENAME = "{basedir}/{basename}_img.png".format(basedir=BASEDIR, basename=TEST_BASE_NAME)
    TEST_MASK_FILENAME =  "{basedir}/{basename}_mask.png".format(basedir=BASEDIR, basename=TEST_BASE_NAME)
    calculate_fingerprint(TEST_IMAGE_FILENAME, TEST_MASK_FILENAME)
    
calculate_from_id(11)
calculate_from_id(15)
calculate_from_id(81)
calculate_from_id(83)