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


# TODO remove ex_id from parameters. rely solely on experiment
def summarize(ex_id, experiment=None, verbose=False, callback=None):
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
        def cb_pri_steps(p, step, num_steps):
            cb_pri((step + p) / num_steps)

    else:
        cb_load = cb_pri = cb_sec = cb_pri_steps = None

    print('preparing blob files')
    if experiment is None:
        # load experiment
        experiment = wio.Experiment(experiment_id=ex_id, callback=cb_load)
        talk('Loaded experiment ID: {}'.format(experiment.id))


    def save_processed_data(data, experiment):
        talk(' - Saving to CSVs...')
        print(' - Saving to CSVs...')
        dumped_keys = []
        for key, value in six.iteritems(data):
            talk('   - {}'.format(key))
            print('   - {}'.format(key))
            experiment.prepdata.dump(data_type=key, dataframe=value, index=False)

        # free up memory once this is saved
        for key in dumped_keys:
            del data[key]

    # process the basic blob data
    talk(' - Summarizing raw data...')
    data = {}
    
    for i, df_type in enumerate(['bounds', 'terminals', 'sizes']):
        if callback:
            cb = lambda x: cb_pri_steps(x, i, 3)
        else:
            cb = None
        print(' - Summarizing {df} data...'.format(df=df_type))
        data[df_type] = primary.create_primary_df(experiment, df_type, callback=cb)
        save_processed_data(data, experiment)

    # TODO: remove this commented method. it keeps failing.
    # data = primary.summarize(experiment, callback=cb_pri)

    # generate secondary data
    talk(' - Generating secondary data...')
    print(' - Generating secondary data...')
    # data['roi'] = secondary.in_roi(experiment=experiment, bounds=data['bounds'])
    data['roi'] = secondary.in_roi(experiment=experiment, bounds=None)
    if callback:
        cb_sec(0.4)
    save_processed_data(data, experiment)
    if callback:
        cb_sec(0.6)

    # data['moved'] = secondary.bodylengths_moved(bounds=data['bounds'], sizes=data['sizes'])
    data['moved'] = secondary.bodylengths_moved(experiment=experiment)
    if callback:
        cb_sec(0.8)
    save_processed_data(data, experiment)
    if callback:
        cb_sec(1)

    # dump it out