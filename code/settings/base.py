'''
File: settings.py
Author: Peter Winter
Description:
'''
# standard imports
import os

# function to give helpful environmental variable import error messages
def get_env_variable(var_name):
    msg = 'Error: set {var} in your environmental settings'.format(var=var_name)
    try:
        return os.environ[var_name]
    except KeyError:
        print(msg)

#set relative paths.
PROJECT_HOME = os.path.dirname(os.path.realpath(__file__)) + '/../../'

MONGO = {
    'use_mongo': False,
    'ip': 'chicago.chem-eng.northwestern.edu',
    'port': 27017,
    'user':'mongo_user',
    'password':'mongo_password',
    'database': 'worm_test',
    # collections
    # are these really necessary?
    'blobs': 'blobs',
    'worms': 'worms',
    'plates': 'plates'
    }

LOGISTICS = {
    'time-series-file-type': 'hdf5', # supports 'hdf5' or 'json'
    # directories for annotation and organization
    'data': PROJECT_HOME + 'data/',
    'worms': PROJECT_HOME + 'data/worms/',
    'plates': PROJECT_HOME + 'data/plates/',
    'dsets': PROJECT_HOME + 'data/dsets/',
    'results': PROJECT_HOME + 'results/',
    'validation':PROJECT_HOME + 'data/annotation/validation/',
    'inventory': PROJECT_HOME + 'data/annotation/inventory/',
    'annotation': PROJECT_HOME + 'data/annotation/experiment_index',
    'pretreatment': PROJECT_HOME + 'data/annotation/pretreatment/',

    'scaling-factors': PROJECT_HOME + 'data/annotation/scaling_factor_images',
    'export': PROJECT_HOME + 'data/export/',

    # filesystem variables
    'filesystem_address': 'peterwinter@barcelona.chem-eng.northwestern.edu',
    'filesystem_data': '/home/projects/worm_movement/Data/MWT_RawData/',
    'filesystem_inventory': '/home/projects/worm_movement/Data/Inventory/',

    # cluster variables
    'cluster_data': '/home/projects/worm_movement/Data/MWT_RawData/',
    'cluster_address': 'peterwinter@phoenix.research.northwestern.edu',
    'qsub_directory': PROJECT_HOME + 'data/qsub/',
    }

SPREADSHEET = {'user': 'your gmail name',
               'password': 'your gmail password',
               # names of spreadsheets
               'spreadsheet': 'annotation',
               'scaling-factors': 'scaling-factors',
               'row-id': 'ex-id',
               'columns': ['dataset', 'label', 'sublabel', 'vid-flags', 'name',
                           'strain', 'age', 'plate-id', 'l1-arrest',
                           'growth-conditions',
                           'set-temp', 'stimulus', 'food', 'compounds',
                           'pixels-per-mm', 'vid-duration', 'num-blobs-files',
                           'num-images', 'source-camera']
               }

FILTER = {'min_body_lengths': 2, 'min_duration': 120, 'min_size': 50}

SMOOTHING = {'spine_order': 5, 'spine_window': 13,
              'time_order': 5, 'time_window': 71,
              'N_points':50, 'time_step':0.1}

JOINING = {'method': 'tapeworm' #None
}

