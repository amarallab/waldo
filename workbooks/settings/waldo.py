import sys
import pathlib

if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
    MWT_DATA_ROOT = pathlib.Path('/home/projects/worm_movement/Data/MWT_RawData')
else:
    MWT_DATA_ROOT = pathlib.Path() / '..' / 'data' / 'mwt'
#MWT_DATA_ROOT = pathlib.Path() / '..' / 'Waldo' / 'data' / 'mwt'

try:
    from multiworm_local import *
except ImportError:
    pass

try:
    from waldo_local import *
except ImportError:
    pass
