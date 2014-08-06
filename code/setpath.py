import sys
import os
from os.path import dirname, join
sys.path.append(join(dirname(__file__))) # already in by default
sys.path.append(join(dirname(__file__), 'shared'))
sys.path.append(join(dirname(__file__), 'shared', 'joining'))

os.environ.setdefault('MULTIWORM_SETTINGS', 'settings.multiworm')
os.environ.setdefault('WALDO_SETTINGS', 'settings.waldo')