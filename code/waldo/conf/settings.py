from __future__ import absolute_import, print_function

import os
import errno
import json

from .defaults import defaults

def _default_user_config():
    try:
        from win32com.shell import shellcon, shell
        homedir = shell.SHGetFolderPath(0, shellcon.CSIDL_APPDATA, 0, 0)
    except ImportError: # quick semi-nasty fallback for non-windows/win32com case
        homedir = os.path.expanduser("~")

    return os.path.join(homedir, "waldo_config.ini")

class Settings(object):
    def __init__(self):
        self.config_file = _default_user_config()
        self._load_defaults()

    def __getattr__(self, name):
        if name in defaults:
            return self._data[name]
        else:
            return super(Settings, self).__getattribute__(name)

    def __setattr__(self, name, value):
        if name in defaults:
            self._data[name] = value
        else:
            super(Settings, self).__setattr__(name, value)

    def _get_or_update_config_file(self, config_file):
        if config_file is not None:
            self.config_file = config_file
        return self.config_file

    def _load_defaults(self):
        self._data = defaults.copy()

    def save(self, config_file=None):
        config_file = self._get_or_update_config_file(config_file)
        with open(config_file, 'w') as f:
            json.dump(self._data, f, indent=4, sort_keys=True)

    def load(self, config_file=None, clean=False, autogenerate=False):
        config_file = self._get_or_update_config_file(config_file)
        try:
            with open(config_file) as f:
                newdata = json.load(f)
        except (IOError, OSError) as e:
            if e.errno == errno.ENOENT and autogenerate:
                self._load_defaults()
                self.save()
                return
            else:
                raise
        if clean:
            self._load_defaults()

        self._data.update(newdata)

settings = Settings()
settings.load(autogenerate=True)
