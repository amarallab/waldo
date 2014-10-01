from __future__ import absolute_import
from six.moves import range

from PIL import Image, ImageOps, ImageChops

from ..tools import Box

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
