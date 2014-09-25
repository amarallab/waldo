from __future__ import (
        absolute_import, division, print_function, unicode_literals)

# standard library

# third party

# package specific
from waldo.conf import settings

from .degree_one import remove_single_descendents, remove_offshoots
from .fission_fusion import remove_fission_fusion, remove_fission_fusion_rel
from .assimilator import assimilate

__all__ = [
    'removal_suite',
]

SUITE_DEFAULTS = {
    'offshoots': settings.COLLIDER_SUITE_OFFSHOOT,
    'splits_abs': settings.COLLIDER_SUITE_SPLIT_ABS,
    'splits_rel': settings.COLLIDER_SUITE_SPLIT_REL,
    'assimilate': settings.COLLIDER_SUITE_ASSIMILATE_SIZE,
}

def removal_suite(digraph, **params):
    params_local = SUITE_DEFAULTS.copy()
    params_local.update(params)

    digraph.validate()

    assimilate(digraph, max_threshold=params_local['assimilate'])
    remove_single_descendents(digraph)

    remove_fission_fusion(digraph, max_split_frames=params_local['splits_abs'])
    remove_fission_fusion_rel(digraph, split_rel_time=params_local['splits_rel'])

    remove_offshoots(digraph, threshold=params_local['offshoots'])
    remove_single_descendents(digraph)
