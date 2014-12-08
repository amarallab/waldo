from __future__ import absolute_import

import os
import json

# COLLIDER
#----------

# Blobs with fewer frames than this are rolled up into their parent
COLLIDER_SUITE_OFFSHOOT = 20
COLLIDER_SUITE_OFFSHOOT_RANGE = (0, 100)   # Range used in the GUI edit

# Blobs that split then rejoin for fewer frames than this are collapsed
COLLIDER_SUITE_SPLIT_ABS = 5
COLLIDER_SUITE_SPLIT_ABS_RANGE = (0, 10)   # Range used in the GUI edit

# Blobs that are split for proportionally less time than this relative
# to the average duration of their parent and child blobs
COLLIDER_SUITE_SPLIT_REL = 0.25
COLLIDER_SUITE_SPLIT_REL_RANGE = (-1, 1, 2)   # Range used in the GUI edit (Double!)

# Blobs that have a duration shorter than this are collapsed into their
# parent and/or child.
COLLIDER_SUITE_ASSIMILATE_SIZE = 10
COLLIDER_SUITE_ASSIMILATE_SIZE_RANGE = (0, 10)   # Range used in the GUI edit

# DEBUG
#-------

DEBUG = False

# LOGGING
#---------

LOG_CONFIGURATION = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'WARN',
            'class':'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'WARN',
            'propagate': True
        }
    }
}

# TAPE
#------

# How the Taper defines a "good blob"
TAPE_REL_MOVE_THRESHOLD = 0.5
TAPE_REL_MOVE_THRESHOLD_RANGE = (0, 1, 2)

# The minimum number of good blob traces to allow before outright failing
TAPE_MIN_TRACE_FAIL = 1
TAPE_MIN_TRACE_FAIL_RANGE = (1, 100)

# The minimum number of good blob traces to allow before warning the user
TAPE_MIN_TRACE_WARN = 30
TAPE_MIN_TRACE_WARN_RANGE = (1, 100)

# Use this many traces to generate the scoring function.  "None" implies no
# limit.
TAPE_TRACE_LIMIT_NUM = 400
TAPE_TRACE_LIMIT_NUM_RANGE = (1, 1000)

# Only search for connections this many frames out
TAPE_FRAME_SEARCH_LIMIT = 600
TAPE_FRAME_SEARCH_LIMIT_RANGE = (1, 1000)

# Take this many samples in the KDE to generate a scoring function
TAPE_KDE_SAMPLES = 100
TAPE_KDE_SAMPLES_RANGE = (1, 300)

# Factor to increase the search cone's slope (maximum speed)
TAPE_MAX_SPEED_MULTIPLIER = 1.50
TAPE_MAX_SPEED_MULTIPLIER_RANGE = (0, 5, 2)

# Pixels added to the radius of the search cone to account for the
# jagged nature of the tracks
TAPE_SHAKYCAM_ALLOWANCE = 10
TAPE_SHAKYCAM_ALLOWANCE_RANGE = (1, 50)

# Moving average window size to filter speeds by
TAPE_MAX_SPEED_SMOOTHING = 10
TAPE_MAX_SPEED_SMOOTHING_RANGE = (1, 100)

# THINGS
#--------
MWT_DATA_ROOT = '/home/projects/worm_movement/Data/MWT_RawData/'
PROJECT_DATA_ROOT = '/Users/heltena/Desktop/waldo/waldo_data'


def _config_filename():
    try:
        from win32com.shell import shellcon, shell
        homedir = shell.SHGetFolderPath(0, shellcon.CSIDL_APPDATA, 0, 0)
    except ImportError: # quick semi-nasty fallback for non-windows/win32com case
        homedir = os.path.expanduser("~")
        return os.path.join(homedir, "waldo_config.ini")

def load():
    try:
        global COLLIDER_SUITE_OFFSHOOT
        global COLLIDER_SUITE_SPLIT_ABS
        global COLLIDER_SUITE_SPLIT_REL
        global COLLIDER_SUITE_ASSIMILATE_SIZE

        global DEBUG

        global TAPE_REL_MOVE_THRESHOLD
        global TAPE_MIN_TRACE_FAIL
        global TAPE_MIN_TRACE_WARN
        global TAPE_TRACE_LIMIT_NUM
        global TAPE_FRAME_SEARCH_LIMIT
        global TAPE_KDE_SAMPLES
        global TAPE_MAX_SPEED_MULTIPLIER
        global TAPE_SHAKYCAM_ALLOWANCE
        global TAPE_MAX_SPEED_SMOOTHING

        global MWT_DATA_ROOT
        global PROJECT_DATA_ROOT

        with open(_config_filename(), "rt") as f:
            data = json.load(f)

        COLLIDER_SUITE_OFFSHOOT = data.get('COLLIDER_SUITE_OFFSHOOT', COLLIDER_SUITE_OFFSHOOT)
        COLLIDER_SUITE_SPLIT_ABS = data.get('COLLIDER_SUITE_SPLIT_ABS', COLLIDER_SUITE_SPLIT_ABS)
        COLLIDER_SUITE_SPLIT_REL = data.get('COLLIDER_SUITE_SPLIT_REL', COLLIDER_SUITE_SPLIT_REL)
        COLLIDER_SUITE_ASSIMILATE_SIZE = data.get('COLLIDER_SUITE_ASSIMILATE_SIZE', COLLIDER_SUITE_ASSIMILATE_SIZE)

        DEBUG = data.get('DEBUG', DEBUG)

        TAPE_REL_MOVE_THRESHOLD = data.get('TAPE_REL_MOVE_THRESHOLD', TAPE_REL_MOVE_THRESHOLD)
        TAPE_MIN_TRACE_FAIL = data.get('TAPE_MIN_TRACE_FAIL', TAPE_MIN_TRACE_FAIL)
        TAPE_MIN_TRACE_WARN = data.get('TAPE_MIN_TRACE_WARN', TAPE_MIN_TRACE_WARN)
        TAPE_TRACE_LIMIT_NUM = data.get('TAPE_TRACE_LIMIT_NUM', TAPE_TRACE_LIMIT_NUM)
        TAPE_FRAME_SEARCH_LIMIT = data.get('TAPE_FRAME_SEARCH_LIMIT', TAPE_FRAME_SEARCH_LIMIT)
        TAPE_KDE_SAMPLES = data.get('TAPE_KDE_SAMPLES', TAPE_KDE_SAMPLES)
        TAPE_MAX_SPEED_MULTIPLIER = data.get('TAPE_MAX_SPEED_MULTIPLIER', TAPE_MAX_SPEED_MULTIPLIER)
        TAPE_SHAKYCAM_ALLOWANCE = data.get('TAPE_SHAKYCAM_ALLOWANCE', TAPE_SHAKYCAM_ALLOWANCE)
        TAPE_MAX_SPEED_SMOOTHING = data.get('TAPE_MAX_SPEED_SMOOTHING', TAPE_MAX_SPEED_SMOOTHING)

        MWT_DATA_ROOT = data.get('MWT_DATA_ROOT', MWT_DATA_ROOT)
        PROJECT_DATA_ROOT = data.get('PROJECT_DATA_ROOT', PROJECT_DATA_ROOT)
        return True
    except Exception as e:
        print "E: Cannot load data.", e.message
        return False

def save():
    try:
        data = {
            'COLLIDER_SUITE_OFFSHOOT': COLLIDER_SUITE_OFFSHOOT,
            'COLLIDER_SUITE_SPLIT_ABS': COLLIDER_SUITE_SPLIT_ABS,
            'COLLIDER_SUITE_SPLIT_REL': COLLIDER_SUITE_SPLIT_REL,
            'COLLIDER_SUITE_ASSIMILATE_SIZE': COLLIDER_SUITE_ASSIMILATE_SIZE,

            'DEBUG': DEBUG,

            'TAPE_REL_MOVE_THRESHOLD': TAPE_REL_MOVE_THRESHOLD,
            'TAPE_MIN_TRACE_FAIL': TAPE_MIN_TRACE_FAIL,
            'TAPE_MIN_TRACE_WARN': TAPE_MIN_TRACE_WARN,
            'TAPE_TRACE_LIMIT_NUM': TAPE_TRACE_LIMIT_NUM,
            'TAPE_FRAME_SEARCH_LIMIT': TAPE_FRAME_SEARCH_LIMIT,
            'TAPE_KDE_SAMPLES': TAPE_KDE_SAMPLES,
            'TAPE_MAX_SPEED_MULTIPLIER': TAPE_MAX_SPEED_MULTIPLIER,
            'TAPE_SHAKYCAM_ALLOWANCE': TAPE_SHAKYCAM_ALLOWANCE,
            'TAPE_MAX_SPEED_SMOOTHING': TAPE_MAX_SPEED_SMOOTHING,

            'MWT_DATA_ROOT': MWT_DATA_ROOT,
            'PROJECT_DATA_ROOT': PROJECT_DATA_ROOT,
        }
        with open(_config_filename(), "wt") as f:
            json.dump(data, f, indent=4, sort_keys=True)
        return True
    except Exception, e:
        print "E: Cannot save data.", e.message
        return False


load()












# # -*- coding: utf-8 -*-
# """
# Waldo configuration management.  Inspired by Django's settings module.  To
# override defaults, specify the module you want to superscede by setting (or
# setdefault) the WALDO_SETTINGS environment variable.  The value of the
# envvar must be importable.
#
#     os.environ.setdefault('WALDO_SETTINGS', 'my_config')
# """
# from __future__ import (
#         absolute_import, division, print_function, unicode_literals)
# import six
# from six.moves import (zip, filter, map, reduce, input, range)
#
# # standard library
# import os
# import importlib #2.7+
# import warnings
# import logging.config
#
# # third party
#
# # package specific
# from . import defaults
#
# ENVIRONMENT_VARIABLE = 'WALDO_SETTINGS'
#
# class Settings(object):
#     def __init__(self, local_module_name):
#         # copy attributes from the default settings
#         for field in dir(defaults):
#             if field.isupper():
#                 setattr(self, field, getattr(defaults, field))
#
#         self.LOCAL_MODULE = local_module_name
#
#         if local_module_name is None:
#             raise EnvironmentError("{} environmental variable "
#                 "not specified.  Set the variable before importing Waldo "
#                 "packages to configure local settings."
#                 .format(ENVIRONMENT_VARIABLE))
#
#         # attempt to import
#         try:
#             local_module = importlib.import_module(local_module_name)
#         except ImportError:
#             raise ImportError("Failed to load settings module: {}"
#                               .format(local_module_name))
#
#         # overwrite attributes loaded from default with local settings
#         rogue_fields = []
#         for field in dir(local_module):
#             if field.isupper():
#                 if not hasattr(self, field):
#                     # see below
#                     rogue_fields.append(field)
#                 setattr(self, field, getattr(local_module, field))
#
#         # notify users if there is an unexpected setting. could be either a
#         # typo, or a "local-only" setting which may result in code that works
#         # locally, but will crash when internal code attempts to reference a
#         # non-existent attribute (gonna have a bad time).
#         if rogue_fields:
#             raise EnvironmentError("Local settings included field(s): {} "
#                     "which have no defaults.  Possible typo?"
#                     .format(', '.join("'{}'".format(f) for f in rogue_fields)))
#
# local_settings = os.environ.get(ENVIRONMENT_VARIABLE)
# settings = Settings(local_settings)
#
# logging.config.dictConfig(settings.LOG_CONFIGURATION)
