__author__ = 'peterwinter'


def write_pathological_input(pathological_input, input_type='', note='', savename=''):
    '''
    this saves any outlines that break my code in order to fix any mistakes.
    '''
    import json
    import time
    import os
    import numpy
    if not savename:
        savename = __file__.split(os.path.basename(__file__))[0] + 'broken_outline_' + str(time.time()) + '.json'
    '''
    x = type(pathological_input)
    if x not in [list, dict, str]:
        print x
    '''
    simplified_input = []
    if hasattr(pathological_input,'__iter__'):
        for i in pathological_input:
            if isinstance(i, numpy.ndarray):
                simplified_input.append(i.tolist())
            else:
                simplified_input.append(i)

    json.dump({'pathological_input': simplified_input, 'input_type':input_type, 'note': note}, open(savename, 'w'))
