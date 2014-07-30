from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library

# third party

# project specific
from conf import settings
import wio
from . import secondary
from . import primary

# import os
# import errno
# import pathlib
# import pickle

# import numpy as np
# import pandas as pd



__all__ = ['summarize']

DATA_DIR = settings.LOGISTICS['filesystem_data']


import numpy as np
import pandas as pd


def preprocess_experiment(ex_id):
    """
    Process a wio.Experiment object using the raw data to generate
    intermediate summary data.
    """
    # load experiment
    experiment = wio.Experiment(experiment_id=ex_id)

    # process the basic blob data
    data = primary.summarize(experiment)

    # generate secondary data
    data['roi'] = secondary.in_roi(experiment=experiment, bounds=data['bounds'])
    data['moved'] = secondary.bodylengths_moved(bounds=data['bounds'], sizes=data['sizes'])

    # save it out
    for key, value in six.iteritems(data):
        experiment.prep_data.dump(data_type=key, dataframe=value, index=False)

    roi = check_roi(ex_id, bounds=bounds)
    prep_data.dump(data_type='roi', dataframe=roi, index=False)

    moved = bodylengths_moved(ex_id, bounds=bounds, sizes=sizes)
    prep_data.dump(data_type='moved', dataframe=moved, index=False)

    return bounds, terminals, sizes
