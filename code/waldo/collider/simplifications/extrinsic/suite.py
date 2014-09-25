from __future__ import (
        absolute_import, division, print_function, unicode_literals)

from .blank_nodes import remove_blank_nodes
from .roi import remove_nodes_outside_roi

__all__ = ['extrinsic_removal_suite']

SUITE_DEFAULTS = {}

def trim_unwanted(graph, experiment):
    remove_nodes_outside_roi(graph, experiment)
    remove_blank_nodes(graph, experiment)

def extrinsic_removal_suite(digraph, experiment, **params):
    params_local = SUITE_DEFAULTS.copy()
    params_local.update(params)

    # threshold = 2
    # suspects = collider.suspected_collisions(graph, threshold)
    # print(len(suspects), 'suspects found')
    # collider.resolve_collisions(graph, experiment, suspects)
