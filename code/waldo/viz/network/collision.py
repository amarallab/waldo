# -*- coding: utf-8 -*-
"""
MWT collision visualization (for screening)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from waldo.viz import tools as vt

def show_collision(experiment, graph, target, direction='backwards'):
    backwards = direction.startswith('back')

    if backwards:
        others = graph.predecessors(target)
    else:
        others = graph.successors(target)
        raise NotImplementedError()

    blob = experiment[target]
    others = [experiment[other] for other in others]

    blob.df.decode_contour()
    for other in others:
        other.df.decode_contour()

    if backwards:
        blob_contour = vt.get_contour(blob, index='first')
        other_contours = [vt.get_contour(other, index='last') for other in others if not other.empty]

    contours = [blob_contour]
    contours.extend(other_contours)

    bounds, patches = vt.patch_contours(contours)
    collection_pre = PatchCollection(patches[1:], cmap=plt.cm.autumn, alpha=0.3)
    collection_post = PatchCollection(patches[:1], facecolor='green', alpha=0.3)
    collection_pre.set_array(np.linspace(0.3, 0.9, len(patches) - 1))

    f, axs = plt.subplots(ncols=2, sharey=True)
    f.set_size_inches((10, 6))
    #f.subplots_adjust(wspace=None)
    f.tight_layout()

    bounds.grow(20)
    bounds.width = max(120, bounds.width)
    bounds.height = max(180, bounds.height)
    #bounds.shape = 120, 180 # this seems to work well enough?

    image = vt.get_image(experiment, time=blob.born_t)
    vmin, vmax = np.percentile(image, [3, 97])

    collections = [collection_pre, collection_post]
    for ax, coll, blobs, frame in zip(axs, collections, [others, [blob]], [others[0].died_f, blob.born_f]):
        ax.imshow(image.T, cmap=plt.cm.Greys, interpolation='none', vmin=vmin, vmax=vmax)
        ax.add_collection(coll)
        ax.set_xlim(bounds.x)
        ax.set_ylim(bounds.y)
        ax.set_aspect('equal')
        ax.set_title('Blob(s): {}, Frame {}'.format(
                ', '.join(str(b.id) for b in blobs),
                int(frame)))

    return f, axs
