from __future__ import absolute_import, division, print_function
from six.moves import range

import numpy as np
import matplotlib.pyplot as plt

from . import pil

__all__ = [
    'show_image',
    'plot_spacetime',
    'plot_spacetime_rainbow',
]

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

def plot_spacetime(axes, experiment, bounds, **kwargs):
    """
    Plot the images from *experiment* on or bracked by *time* (can be a
    scalar or len-2 sequence) on *axes* (can be one or an iterable of axes
    objects).

    Loose keyword arguments passed to :py:func:`show_image`.
    """
    # find and load images
    temporal_kwas = split_keys(kwargs, ['time', 'frame'])
    if not temporal_kwas:
        raise ValueError("'time' or 'frame' must be provided")

    try:
        image_files = experiment.image_files.spanning(**temporal_kwas)
    except TypeError:
        image_files = [experiment.image_files.nearest(**temporal_kwas)[0]]

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
    try:
        for ax in axes:
            show_image(ax, image, extents, bounds, **kwargs)
    except TypeError:
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

def split_keys(d, keys):
    new_d = {}
    for k in keys:
        try:
            new_d[k] = d.pop(k)
        except KeyError:
            pass
    return new_d

def plot_spacetime_rainbow(axes, experiment, bounds, **kwargs):
    """
    Plot the images from *experiment* on or bracked by *time* or *frame* (can be a
    scalar or len-2 sequence) on *axes* (can be one or an iterable of axes
    objects).

    Keyword Arguments
    -----------------
    time : **required** (or frame)
        scalar or 2-ple of time/frame(s)
    frame : **required** (or time)
        scalar or 2-ple of time/frame(s)

    Loose keyword arguments passed to :py:func:`show_image`.
    """
    # split some kwargs
    temporal_kwas = split_keys(kwargs, ['time', 'frame'])
    if not temporal_kwas:
        raise ValueError("'time' or 'frame' must be provided")
    rainbow_kwas = split_keys(kwargs, ['hue_range'])

    if not isinstance(bounds, pil.Box):
        bounds = pil.Box(bounds)

    # find and load images
    try:
        image_files = experiment.image_files.spanning(**temporal_kwas)
    except TypeError:
        image_files = [experiment.image_files.nearest(**temporal_kwas)[0]]

    images, extents = zip(*(
            pil.load_image_portion(experiment, bounds, filename=fn)
            for fn
            in image_files))

    # flatten time
    if len(images) > 1:
        image = pil.rainbow_merge(images, **rainbow_kwas)
        assert all(sum(extents) == e for e in extents)
    else:
        image = images[0]

    extents = extents[0]

    # plot
    try:
        for ax in axes:
            show_image(ax, image, extents, bounds, **kwargs)
    except TypeError:
        show_image(axes, image, extents, bounds, **kwargs)
