import pathlib

# NOTE: MAKES ASSUMPTION ABOUT PROJECT STRUCTURE AND WHERE THIS FILE IS
PROJECT_HOME = str((pathlib.Path(__file__).parent / '..' / '..' / '..').resolve())

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
    'results': PROJECT_HOME + 'results/',

    'data': PROJECT_HOME + 'data/',

    'export': PROJECT_HOME + 'data/export/',
    'worms': PROJECT_HOME + 'data/worms/',
    'plates': PROJECT_HOME + 'data/plates/',
    'dsets': PROJECT_HOME + 'data/dsets/',
    'prep': PROJECT_HOME + 'data/prep',
    #'nodenotes': PROJECT_HOME + 'data/prep/nodenotes/',
    #'matches': PROJECT_HOME + 'data/prep/matches/',
    #'accuracy': PROJECT_HOME + 'data/prep/acuracy/',
    #'validation':PROJECT_HOME + 'data/annotation/validation/',
    'inventory': PROJECT_HOME + 'data/annotation/inventory/',
    'annotation': PROJECT_HOME + 'data/annotation/experiment_index',
    'scaling-factors': PROJECT_HOME + 'data/annotation/scaling_factor_images',


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
