'''
File: settings.py
Author: Peter Winter
Description:
'''

# import everything from settings.base and overwrite what is necessary
import base
import sys

# testing sheets
base.SPREADSHEET['scaling-factors'] = 'testsf'

configured = False

def configure(configuration_name=None):
    if configuration_name is None:
        print "Default configuration."
    else:
        print "Configuring name: '%s'" % configuration_name

    if configuration_name is None:
        pass

    elif configuration_name == 'heltena':
        # passwords
        base.MONGO['user'] = base.get_env_variable('WORM_MONGO_USER')
        base.MONGO['password'] = base.get_env_variable('WORM_MONGO_PASSWORD')
        base.SPREADSHEET['user'] = base.get_env_variable('WORM_SPREADSHEET_USER')
        base.SPREADSHEET['password'] = base.get_env_variable('WORM_SPREADSHEET_PASSWORD')

        base.LOGISTICS['filesystem_data'] = '/Users/heltena/src/waldo/data/heltena/'

    elif configuration_name == 'peterwinter':
        # passwords
        base.MONGO['user'] = base.get_env_variable('WORM_MONGO_USER')
        base.MONGO['password'] = base.get_env_variable('WORM_MONGO_PASSWORD')
        base.SPREADSHEET['user'] = base.get_env_variable('WORM_SPREADSHEET_USER')
        base.SPREADSHEET['password'] = base.get_env_variable('WORM_SPREADSHEET_PASSWORD')

    global configured
    configured = True

try:
    index = sys.argv.index('-c')
    configure_name = sys.argv[index + 1]
    sys.argv.pop(index)
    sys.argv.pop(index)
    configure(configure_name)
except ValueError:
    configure() # default

get_env_variable = base.get_env_variable
PROJECT_DIRECTORY = base.PROJECT_DIRECTORY
MONGO = base.MONGO
LOGISTICS = base.LOGISTICS
SPREADSHEET = base.SPREADSHEET
FILTER = base.FILTER
SMOOTHING = base.SMOOTHING
