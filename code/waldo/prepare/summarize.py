from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import functools

# third party

# project specific
from waldo.conf import settings
from waldo import wio
from . import secondary
from . import primary

__all__ = ['summarize']

CALLBACK_LOAD_FRAC = 0.02
CALLBACK_PRIMARY_FRAC = 0.90
CALLBACK_SECONDARY_FRAC = 0.08

def summarize(ex_id, verbose=False, callback=None):
    """
    intermediate summary data.
    """
    if verbose:
        talk = print
    else:
        talk = lambda *a, **k: None

    if callback:
        def cb_load(p):
            callback(CALLBACK_LOAD_FRAC * p)
        def cb_pri(p):
            callback(CALLBACK_LOAD_FRAC + CALLBACK_PRIMARY_FRAC * p)
        def cb_sec(p):
            callback(CALLBACK_LOAD_FRAC + CALLBACK_PRIMARY_FRAC +
                    CALLBACK_SECONDARY_FRAC * p)
    else:
        cb_load = cb_pri = cb_sec = None

    # load experiment
    experiment = wio.Experiment(experiment_id=ex_id, callback=cb_load)
    talk('Loaded experiment ID: {}'.format(experiment.id))

    # process the basic blob data
    talk(' - Summarizing raw data...')
    data = primary.summarize(experiment, callback=cb_pri)

    # generate secondary data
    talk(' - Generating secondary data...')
    data['roi'] = secondary.in_roi(experiment=experiment, bounds=data['bounds'])
    if callback:
        cb_sec(0.25)

    data['moved'] = secondary.bodylengths_moved(bounds=data['bounds'], sizes=data['sizes'])
    if callback:
        cb_sec(0.5)

    # dump it out
    talk(' - Dumping to CSVs...')
    for key, value in six.iteritems(data):
        print(key, type(value))
        talk('   - {}'.format(key))
        experiment.prepdata.dump(data_type=key, dataframe=value, index=False)
    if callback:
        cb_sec(1)
