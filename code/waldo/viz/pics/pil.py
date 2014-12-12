from __future__ import absolute_import, division, print_function
from six.moves import range

import collections

import numpy as np
import matplotlib.colors as colors
from PIL import Image, ImageOps, ImageChops

from ..tools import Box

__all__ = [
    'merge_stack',
    'rainbow_merge',
]

def pillowed(img_file):
    """
    PIL load...haha.

    Transposes the image into the correct coordinate system
    """
    img = Image.open(img_file)
    img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
    return img

load = pillowed # ok, not that funny.

def crop(img, box):
    """
    Crops the given PIL image around the given box (dimensions must be
    integers to align with pixels). Returns another PIL image and actual
    extent box (if box exceeds image borders)
    """
    extents = Box(box)
    width, height = img.size
    extents.bottom = max(0, extents.bottom)
    extents.left = max(0, extents.left)
    extents.top = min(height, extents.top)
    extents.right = min(width, extents.right)

    retimg = img.crop(extents.PIL)
    extents.center = (c - 0.5 for c in extents.center)
    return retimg, extents

def adjust(im):
    # - reduce 16-bit greyscale to 8-bit
    im = im.point([int(x/256) for x in range(2**16)], 'L')
    im = ImageOps.autocontrast(im, cutoff=0)
    return im

def merge_stack(images):
    '''
    Merges a series of *images* using ***math***.

    Parameters
    ----------
    images : iterable of :py:class:`PIL.Image.Image`
        Image stack

    Returns
    -------
    :py:class:`PIL.Image.Image`
        Merged image
    '''
    # Guards
    if not images:
        raise ValueError('No images provided')
    elif len(images) == 1:
        return images[0]

    # Convert Image
    # - compress first
    #print images[0].histogram()
    #lut = equalize(images[0].histogram())

    # stack up, keeping darkest pixels
    composite = ImageChops.darker(images[0], images[1])
    for im in images[2:]:
        composite = ImageChops.darker(composite, im)
    return composite

def red_on_black(im, invert=True):
    """
    Convert the greyscale *im* into an RGB image where bright = red,
    dark = black (only the red channel is used, flipped if *invert* is
    ``False``)
    """
    assert im.mode == 'L'
    if invert:
        im = ImageOps.invert(im)
    fill = Image.new('L', im.size, 0)
    im = Image.merge('RGB', (im, fill, fill))
    return im

def red_on_white(im):
    """
    Convert the greyscale *im* into an RGB image where bright = white,
    dark = red.
    """
    assert im.mode == 'L'
    fill = Image.new('L', im.size, 255)
    im = Image.merge('RGB', (fill, im, im))
    return im

def rotate_hue(im, x):
    """
    Adjust the hue of *im* by *x*, where *x* is a value between 0.0 to 1.0.
    Full red to full green is 1/3, red to blue is 2/3.
    """
    assert im.mode == 'RGB'
    ima = np.asarray(im) / 255

    ima_hsv = colors.rgb_to_hsv(ima)
    ima_hsv[...,0] = (ima_hsv[...,0] + x) % 1
    ima = colors.hsv_to_rgb(ima_hsv)

    ima *= 255
    ima = ima.astype(np.uint8)

    im = Image.fromarray(ima)
    return im

def rainbow_merge(images, hue_range=(2/3, 0), inverted=False):
    '''
    Merges a series of *images* using ***unicorns***.

    Parameters
    ----------
    images : sequence of :py:class:`PIL.Image.Image`
        Image stack

    Keyword Arguments
    -----------------
    hue_range : sequence (length 2)
        2-ple of hue range. Default of (2/3, 0) goes from blue to red.
    inverted : boolean
        Show empty background as dark?

    Returns
    -------
    :py:class:`PIL.Image.Image`
        Merged image
    '''
    # Guards
    if not images:
        raise ValueError('No images provided')
    elif not isinstance(images, collections.Sequence):
        raise TypeError('Must be a sequence')

    if inverted:
        mixer = ImageChops.lighter
        colorer = red_on_black
    else:
        mixer = ImageChops.darker
        colorer = red_on_white

    n_images = len(images)
    if n_images == 1:
        # make an identical start and end
        images = [images[0]] * 2
        n_images = 2

    def hue_adj(i):
        return hue_range[0] + (hue_range[1] - hue_range[0]) * (i / n_images)

    # stack up, keeping darkest pixels
    images = iter(images)
    composite = None
    for i, image in enumerate(images):
        image = colorer(image)
        image = rotate_hue(image, hue_adj(i))
        if composite is None:
            composite = image
            continue
        composite = mixer(composite, image)

    return composite

def load_image_portion(experiment, bounds, **time_frame_or_filename):
    """
    Load the nearest image in time from the experiment, crop, and adjust
    """
    if 'filename' in time_frame_or_filename:
        filename = time_frame_or_filename['filename']
    else:
        filename = experiment.image_files.nearest(**time_frame_or_filename)[0]

    img, extents = crop(load(str(filename)), bounds)
    img = adjust(img)

    return img, extents
