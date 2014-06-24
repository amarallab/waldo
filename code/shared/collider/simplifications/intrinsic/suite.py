from __future__ import (
        absolute_import, division, print_function, unicode_literals)

from .degree_one import remove_single_descendents, remove_offshoots
from .fission_fusion import remove_fission_fusion, remove_fission_fusion_rel

__all__ = [
    'removal_suite',
]

SUITE_DEFAULTS = {
    'offshoots': 20,
    'splits_abs': 5,
    'splits_rel': 0.5,
}

def removal_suite(digraph, **params):
    params_local = SUITE_DEFAULTS.copy()
    params_local.update(params)

    remove_single_descendents(digraph)
    remove_fission_fusion(digraph, max_split_frames=params_local['splits_abs'])
    remove_fission_fusion_rel(digraph, split_rel_time=params_local['splits_rel'])
    remove_offshoots(digraph, threshold=params_local['offshoots'])
    remove_single_descendents(digraph)
