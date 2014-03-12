'''
File: settings.py
Author: Peter Winter
Description:
'''

# import everything from settings.base and overwrite what is necessary
from base import *

import getpass

# testing sheets
SPREADSHEET['spreadsheet'] = 'testan'
SPREADSHEET['scaling-factors'] = 'testsf'

username = getpass.getuser()
if username == 'heltena':
    # passwords
    MONGO['user'] = get_env_variable('WORM_MONGO_USER')
    MONGO['password'] = get_env_variable('WORM_MONGO_PASSWORD')
    SPREADSHEET['user'] = get_env_variable('WORM_SPREADSHEET_USER')
    SPREADSHEET['password'] = get_env_variable('WORM_SPREADSHEET_PASSWORD')

    LOGISTICS['filesystem_data'] = '/Users/heltena/src/waldo/data/heltena/'

elif username == 'peterwinter':
    # passwords
    MONGO['user'] = get_env_variable('WORM_MONGO_USER')
    MONGO['password'] = get_env_variable('WORM_MONGO_PASSWORD')
    SPREADSHEET['user'] = get_env_variable('WORM_SPREADSHEET_USER')
    SPREADSHEET['password'] = get_env_variable('WORM_SPREADSHEET_PASSWORD')

