__author__ = 'heltena'

import os
import json


class ConfigurationData(object):
    def __init__(self):
        try:
            from win32com.shell import shellcon, shell
            self.homedir = shell.SHGetFolderPath(0, shellcon.CSIDL_APPDATA, 0, 0)
        except ImportError: # quick semi-nasty fallback for non-windows/win32com case
            self.homedir = os.path.expanduser("~")

        self.configFileName = os.path.join(self.homedir, "waldo_config.ini")

        try:
            with open(self.configFileName, 'rt') as f:
                self._data = json.load(f)
                if self._data is None:
                    self._data = {}
                else:
                    self._data = {k: v for k, v in self._data.items() if v is not None}
        except (ValueError, IOError) as e:
            self._data = {}

    @property
    def waldo_folder(self):
        if 'waldo_folder' in self._data:
            return self._data['waldo_folder']
        else:
            return os.path.expanduser('~/waldo')

    @waldo_folder.setter
    def waldo_folder(self, value):
        self._data['waldo_folder'] = value

    def save(self):
        try:
            with open(self.configFileName, 'wt') as f:
                json.dump(self._data, f)
            return True
        except IOError:
            print "E: Cannot save data"
            return False
        except Exception, e:
            print "E: Cannot save data. Unknown error. ", e.message
            return False
