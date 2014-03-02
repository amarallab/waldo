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
        print msg

#set relative paths.
PROJECT_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + '/../../'

MONGO = {
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
    # directories for annotation and organization
    'data': PROJECT_DIRECTORY + 'data/',
    'plates': PROJECT_DIRECTORY + 'data/plates/',
    'inventory': PROJECT_DIRECTORY + 'data/annotation/inventory/',
    'annotation': PROJECT_DIRECTORY + 'data/annotation/experiment_index',
    'scaling-factors': PROJECT_DIRECTORY + 'data/annotation/scaling_factor_images',
    'export': PROJECT_DIRECTORY + 'data/export/',

    # filesystem variables
    'filesystem_address': 'peterwinter@barcelona.chem-eng.northwestern.edu',
    'filesystem_data': '/home/projects/worm_movement/Data/MWT_RawData/',

    # cluster variables
    'cluster_data': '/home/projects/worm_movement/Data/MWT_RawData/',
    'cluster_address': 'peterwinter@phoenix.research.northwestern.edu',
    'qsub_directory': PROJECT_DIRECTORY + 'data/qsub/',
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
'''    
'vid-flags', 'name', 'vid-duration',
'num-blobs-files', 'num-images', 'source-camera',
'purpose', 'label', 'strain', 'age', 'growth-medium',
'set-temp', 'stimulus', 'food', 'compounds',
'lid', 'plate-id', 'l1-arrest', 'pixels-per-mm'
'''

FILTER = {'min_body_lengths': 2, 'min_duration': 120, 'min_size': 50}

SMOOTHING = {'spine_order': 5, 'spine_window': 13,
              'time_order': 5, 'time_window': 71,
              'N_points':50, 'time_step':0.1}


