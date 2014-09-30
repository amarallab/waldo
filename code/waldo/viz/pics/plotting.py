from __future__ import absolute_import
from six.moves import range

import numpy as np
import matplotlib.pyplot as plt

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

def show_image(axes, image, extents, limits=None, cmap=plt.cm.YlGn):
    if limits is None:
        limits = extents

    axes.set_aspect('equal')

    axes.fill([limits.left, limits.left, limits.right, limits.right],
              [limits.bottom, limits.top, limits.top, limits.bottom],
        hatch='////', facecolor='0.7', edgecolor='0.9',
        zorder=-10)

    print(extents)
    axes.imshow(np.asarray(image), cmap=cmap, extent=extents.vflip,
              interpolation='nearest')

    axes.set_xlim(limits.x)
    axes.set_ylim(limits.y)
