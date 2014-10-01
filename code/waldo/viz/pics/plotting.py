from __future__ import absolute_import
from six.moves import range

import numpy as np
import matplotlib.pyplot as plt

from . import pil

def tweak_bounds(bounds, padding=40, min_dim=120, aspect=1):
    # modify bounds as desired for the figure
    bounds.grow(padding)
    bounds.adjust_aspect(aspect)
    if bounds.width < min_dim:
        bounds.width = min_dim
    if bounds.height < min_dim:
        bounds.height = min_dim
    bounds.adjust_aspect(aspect)
    bounds.round()
    return bounds

def show_image(axes, image, extents, limits=None, cmap=plt.cm.Greys):
    if limits is None:
        limits = extents

    axes.set_aspect('equal')

    axes.fill([limits.left, limits.left, limits.right, limits.right],
              [limits.bottom, limits.top, limits.top, limits.bottom],
        hatch=r'/-|', facecolor='0.75', edgecolor='0.85',
        zorder=-10)

    axes.imshow(np.asarray(image), cmap=cmap, extent=extents.vflip,
              interpolation='nearest')

    axes.set_xlim(limits.x)
    axes.set_ylim(limits.y)

def plot_spacetime(axes, experiment, bounds, time, **kwargs):
    # find and load images
    try:
        image_files = experiment.image_files.spanning(times=time)
    except TypeError:
        image_files = [experiment.image_files.nearest(time=time)[0]]

    images, extents = zip(*(
            pil.load_image_portion(experiment, bounds, filename=fn)
            for fn
            in image_files))

    # flatten time
    if len(images) > 1:
        image = pil.merge_stack(images)
        assert all(sum(extents) == e for e in extents)
    else:
        image = images[0]

    extents = extents[0]

    # plot
    show_image(axes, image, extents, bounds, **kwargs)

def plot_contour(axes, experiment, bid, frame, **kwargs):
    """
    Plot the contour for the given *bid* from *experiment* at *frame*
    on *axes*.

    *frame* can optionally be ``'first'`` or ``'last'``, rather than a
    number for the first or last valid contour available.  If the contour
    does not exist for the given frame (or not at all), a circle will be
    drawn around the centroid of the blob.

    Keyword Arguments
    -----------------
    alpha : float
        0 = invisible, 1 = opaque. Defaults to 0.7.
    facecolor
        Something recognized by Matplotlib. Defaults to red.
    """
    alpha = kwargs.get('alpha', 0.7)
    facecolor = kwargs.get('facecolor', 'red')

    raise NotImplementedError()
