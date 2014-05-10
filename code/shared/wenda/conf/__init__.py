# -*- coding: utf-8 -*-
"""
Wanda configuration management
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import os
import imp

from . import defaults

local_settings = os.environ.get('WENDA_SETTINGS')

class Settings(object):
    def __init__(self, settings_module):
        # copy the global settings into the object
        for setting in dir(defaults):
            if setting.isupper():
                setattr(self, setting, getattr(defaults, setting))

        self.SETTINGS_MODULE = settings_module

        try:
            file, pathname, desc = imp.find_module(settings_module)
            local_settings = imp.load_module(settings_module, file, pathname, desc)
        except ImportError:
            raise ImportError("Failed to load settings module: {}".format(settings_module))

        for setting in dir(local_settings):
            if setting.isupper():
                setattr(self, setting, getattr(local_settings, setting))


settings = Settings(local_settings)
