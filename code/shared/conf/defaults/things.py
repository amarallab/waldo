import pathlib

# NOTE: MAKES ASSUMPTION ABOUT PROJECT STRUCTURE AND WHERE THIS FILE IS
PROJECT_HOME_pl = (pathlib.Path(__file__).parent / '..'  / '..' / '..' / '..').resolve()
PROJECT_HOME = str(PROJECT_HOME_pl) # down-convert pathlib object to plain string

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
    'results': PROJECT_HOME_pl / 'results',

    'data': PROJECT_HOME_pl / 'data',

    'export': PROJECT_HOME_pl / 'data' / 'export',
    'worms': PROJECT_HOME_pl / 'data' / 'worms',
    'plates': PROJECT_HOME_pl / 'data' / 'plates',
    'dsets': PROJECT_HOME_pl / 'data' / 'dsets',
    'prep': PROJECT_HOME_pl / 'data' / 'prep',
    #'nodenotes': PROJECT_HOME_pl / 'data' / 'prep' / 'nodenotes',
    #'matches': PROJECT_HOME_pl / 'data' / 'prep' / 'matches',
    #'accuracy': PROJECT_HOME_pl / 'data' / 'prep' / 'acuracy',
    #'validation':PROJECT_HOME_pl / 'data' / 'annotation' / 'validation',
    'inventory': PROJECT_HOME_pl / 'data' / 'annotation' / 'inventory',
    'annotation': PROJECT_HOME_pl / 'data' / 'annotation' / 'experiment_index',
    'scaling-factors': PROJECT_HOME_pl / 'data' / 'annotation' / 'scaling_factor_images',

    # filesystem variables
    'filesystem_address': 'peterwinter@barcelona.chem-eng.northwestern.edu',
    'filesystem_data': '/home/projects/worm_movement/Data/MWT_RawData/',
    'filesystem_inventory': '/home/projects/worm_movement/Data/Inventory/',

    # cluster variables
    'cluster_data': '/home/projects/worm_movement/Data/MWT_RawData/',
    'cluster_address': 'peterwinter@phoenix.research.northwestern.edu',
    'qsub_directory': PROJECT_HOME_pl / 'data' / 'qsub',
}

MWT_DATA_ROOT = LOGISTICS['filesystem_data']

# compatibility
for key in LOGISTICS:
    if isinstance(LOGISTICS[key], pathlib.Path):
        LOGISTICS[key] = str(LOGISTICS[key])

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
