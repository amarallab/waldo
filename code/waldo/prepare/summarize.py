from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import os

# third party

# project specific
from waldo.conf import settings
from waldo import wio
from . import secondary
from . import primary

__all__ = ['summarize']

DATA_DIR = settings.LOGISTICS['filesystem_data']

def summarize(ex_id, verbose=False):
    """
    intermediate summary data.
    """
    if verbose:
        talk = print
    else:
        talk = lambda *a, **k: None

    # load experiment
    experiment = wio.Experiment(fullpath=os.path.join(DATA_DIR, ex_id))
    talk('Loaded experiment ID: {}'.format(experiment.id))

    # process the basic blob data
    talk(' - Summarizing raw data...')
    data = primary.summarize(experiment)

    # generate secondary data
    talk(' - Generating secondary data...')
    data['roi'] = secondary.in_roi(experiment=experiment, bounds=data['bounds'])
    data['moved'] = secondary.bodylengths_moved(bounds=data['bounds'], sizes=data['sizes'])

    # dump it out
    talk(' - Dumping to CSVs...')
    for key, value in six.iteritems(data):
        talk('   - {}'.format(key))
        experiment.prepdata.dump(data_type=key, dataframe=value, index=False)
