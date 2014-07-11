import os
from fukuda import C_PATH, compile_cpp

COMPILED_FILES = ['fukuda_breakpoints', 'fukuda_segmentation',
                  'generate_data']

FIlES_ARE_HERE = True
for f in COMPILED_FILES:
    if not os.path.isfile(f):
        FIlES_ARE_HERE = False

compile_cpp()
