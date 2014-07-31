import sys
import os

def add_rel_path(*args):
    sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), *args)))

add_rel_path('..', 'code')
add_rel_path('..', 'code', 'shared')
add_rel_path('..', 'code', 'shared', 'joining')

os.environ.setdefault('MULTIWORM_SETTINGS', 'settings.multiworm')
os.environ.setdefault('WALDO_SETTINGS', 'settings.waldo')
