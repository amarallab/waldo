from distutils.core import setup
import py2exe
import matplotlib
import FileDialog

import os
import glob
import waldo.extern

def find_data_files(source, target, patterns):
    if glob.has_magic(source) or glob.has_magic(target):
        raise ValueError("Magic not allowed in source, target")
    ret = {}
    for pattern in patterns:
        pattern = os.path.join(source, pattern)
        for filename in glob.glob(pattern):
            if os.path.isfile(filename):
                targetpath = os.path.join(target, os.path.relpath(filename, source))
                path = os.path.dirname(targetpath)
                ret.setdefault(path, []).append(filename)
    
    return sorted(ret.items())

data_files = []
data_files.extend(matplotlib.get_py2exe_datafiles())
data_files.extend(find_data_files('C:\\Program Files (x86)\\Graphviz2.38\\bin', '', ['config6', '*.dll', '*.exe']))
data_files.extend(find_data_files('c:\\Python27\\Lib\\site-packages\\skimage\io\_plugins', 'skimage\\io\\_plugins', ['*.ini']))
data_files.extend(find_data_files('c:\\Python27\\lib\\site-packages\\brewer2mpl\data', 'brewer2mpl\\data', ['*.json', '*.txt']))

setup(windows=['guiwaldo.py'],
      data_files=data_files,
      options = {"py2exe": {"skip_archive": True,
                            "packages": ["matplotlib", "pytz", "skimage"],
                            "includes": ["sip",
                                         "h5py.*",
                                         "graphviz",
                                         "skimage.*",
                                         "skimage.io.*",
                                         "PIL",
                                         "skimage.io._plugins.*",
                                         "scipy.sparse.csgraph._validation",
                                         "scipy.special._ufuncs_cxx",
                                         ],
                            #             "tcl"],
                            #"bundle_files": 1,
                            "dll_excludes": ["MSVCP90.dll",
                                             "libgdk-win32-2.0-0.dll",
                                             "libgobject-2.0-0.dll",
                                             "libgdk_pixbuf-2.0-0.dll",
                                             "libgtk-win32-2.0-0.dll",
                                             "libglib-2.0-0.dll",
                                             "libcairo-2.dll",
                                             "libpango-1.0-0.dll",
                                             "libpangowin32-1.0-0.dll",
                                             "libpangocairo-1.0-0.dll",
                                             "libglade-2.0-0.dll",
                                             "libgmodule-2.0-0.dll",
                                             "libgthread-2.0-0.dll",
                                             #"QtGui4.dll",
                                             #"QtCore.dll",
                                             #"QtCore4.dll"
                                             ]
                            }
                 })