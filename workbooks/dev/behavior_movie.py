
import sys
import os

#import itertools
import pathlib

import random
import pandas as pd
import numpy as np
import angles
#import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpltools import style
#from mpltools import layout
#from networkx import Graph
import scipy
import scipy.ndimage.morphology as morph


sys.path.append('..')
import about
import pathcustomize

from waldo.conf import settings
from waldo.wio.experiment import Experiment
from waldo.encoding.decode_outline import decode_outline, encode_outline

from behavior import Behavior_Coding, Worm_Shape

settings.PROJECT_DATA_ROOT = '../../Data/test'
pl = pathlib.Path(settings.PROJECT_DATA_ROOT)

day_1_eids = [  '20130318_131056',
                '20130318_131111',
                '20130318_131113',
                '20130318_142605',
                '20130318_142613',
                '20130318_153741',
                '20130318_153742',
                '20130318_153749',
                '20130318_165642',
                '20130318_165643',
                '20130318_165649']


eid = day_1_eids[0]
print(eid)

path = pl / eid / 'blob_files'

e = Experiment(fullpath=path,experiment_id=eid)
typical_bodylength = e.typical_bodylength
b_list = []
for i, (bid, blob) in enumerate(e.blobs()):
    blob_df = blob.df
    if len(blob_df) > 5000:
        t = blob_df['time']
        if t.iloc[-1] - t.iloc[0] > 40 * 60:
            b_list.append(blob_df)
    if len(b_list) >=3:
        break

blob_df = b_list[2]
print(blob_df)

# bc = Behavior_Coding(bl=typical_bodylength)
# bc.read_from_blob_df(blob_df=blob_df)
# bc.preprocess(dt=0.2)
# bc.reassign_front_back(speed_cuttoff = 1.0, ar_cut= 0.8, min_points=5)

worm_shape = Worm_Shape()
worm_shape.read_blob_df(blob_df)
worm_shape.create_contours()

worm_shape.contours[0]
