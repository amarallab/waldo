import sys
import pathlib

if sys.platform.startswith('linux'):
    MWT_DATA_ROOT = pathlib.Path('/home/projects/worm_movement/Data/MWT_RawData')
else:
    MWT_DATA_ROOT = pathlib.Path() / '..' / 'data' / 'mwt'
#MWT_DATA_ROOT = pathlib.Path() / '..' / 'Waldo' / 'data' / 'mwt'