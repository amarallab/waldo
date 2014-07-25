from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

def calculate_fingerprint(image_filename, mask_filename, output_filename):
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

    MAX_PLUS = 2.0
    MAX_MINUS = 1.0
    MAX_DIST = 80.0
    plus_scores = [MAX_PLUS]
    minus_scores = [MAX_MINUS]
    dist_scores = [MAX_DIST]
    ix, iy = 0, 0
    for i in range(len(image)-1):
        iv = image[i]
        if iv > 0:
            jx, jy = move_pixel(ix, iy)
            for j in range(i+1, len(image)):
                jv = image[j]
                if jv > 0:
                    dist = math.hypot(jx-ix, jy-iy)
                    if dist > MAX_DIST:
                        print "E: dist greater than maximum (%f > %f)" % (dist, MAX_DIST)
                    plus_scores.append(iv + jv)
                    minus_scores.append(abs(iv - jv))
                    dist_scores.append(dist)
                jx, jy = move_pixel(jx, jy)
        ix, iy = move_pixel(ix, iy)

    plus_heatmap, _, _ = np.histogram2d(dist_scores, plus_scores, bins=(32,32))
    minus_heatmap, _, _ = np.histogram2d(dist_scores, minus_scores, bins=(32,32))
    f, ax = plt.subplots(1,2)
    f.set_size_inches((6,3))
    ax[0].imshow(plus_heatmap, vmin=0, vmax=1024)
    ax[1].imshow(minus_heatmap, vmin=0, vmax=1024)
    f.savefig(output_filename)

def main(args):
    for arg in args:
        if not arg.endswith("_img.png"):
            print "E: file '%s' cannot be treated"
        else:
            image_filename = arg
            mask_filename = arg[0:-len("_img.png")] + "_mask.png"
            fingerprint_filename = arg[0:-len("_img.png")] + "_fingerprint.png"
            calculate_fingerprint(image_filename, mask_filename, fingerprint_filename)

if __name__ == "__main__":
    main(sys.argv[1:])
