import os
import sys
from glob import glob
import pickle
import itertools

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import prettyplotlib as ppl



import setpath
os.environ.setdefault('WALDO_SETTINGS', 'default_settings')


from conf import settings
import wio.file_manager as fm
from wio.experiment import Experiment
import tape.taper as tp


DATA_DIR = settings.LOGISTICS['filesystem_data']



ex_id = '20130318_131111'
#path = os.path.join(DATA_DIR, ex_id)
#print path
experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
graph = pickle.load(open('/home/projects/worm_movement/Data/dev/collider_networks/20130318_131111_graphcache2.pkl'))

taper = tp.Taper(experiment=experiment, graph=graph)
s, e = taper.find_start_and_end_nodes()
matches = taper.score_potential_matches(s,e)


#prep_data = fm.PrepData('20130318_131111')
#print prep_data.filedir
#print prep_data.data_types
#terminals = prep_data.load('terminals')
#a = tp.calculate_relations(graph, terminals)
