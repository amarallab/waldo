# -*- coding: utf-8 -*-
"""
WIO2 configuration management.  Inspired by Django's settings module.  To
override defaults, specify the module you want to superscede by setting (or
setdefault) the WIO_SETTINGS environment variable.

    os.environ.setdefault('WIO_SETTINGS', 'my_config')
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import os
import imp

from . import defaults

local_settings = os.environ.get('WIO_SETTINGS')

class Settings(object):
    def __init__(self, settings_module):
        # copy the global settings into the object
        for setting in dir(defaults):
            if setting.isupper():
                setattr(self, setting, getattr(defaults, setting))

        self.SETTINGS_MODULE = settings_module

        if settings_module is not None:
            try:
                file, pathname, desc = imp.find_module(settings_module)
                local_settings = imp.load_module(settings_module, file, pathname, desc)
            except ImportError:
                raise ImportError("Failed to load settings module: {}".format(settings_module))

            for setting in dir(local_settings):
                if setting.isupper():
                    setattr(self, setting, getattr(local_settings, setting))


settings = Settings(local_settings)
