#!/usr/bin/env python

'''
Filename: index_file.py
Description: index .jsons that link each ex_id to a specific property like 'age' or 'source-camera'.

'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import json
import time

# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(SHARED_DIR)
sys.path.append(CODE_DIR)

# nonstandard imports
import WormMetrics.switchboard as sb
import database.mongo_retrieve as mr
from settings.local import LOGISTICS

# globals
OUTDIR = LOGISTICS['export']

'''
def export_ex_id_index_json(props='age', out_dir=OUTDIR, **kwargs):
    """
    writes a json with a dictionary of ex_ids and the value of a correspoinding property
    :param prop: the property to be exported (key string for database documents)
    :param out_dir: the directory in which to save the json.
    """
    save_name = '{d}/ex_id_{prop}.json'.format(d=out_dir, prop=prop)
    data = {}
    for e in mr.mongo_query({'purpose': 'N2_aging', 'data_type': 'smoothed_spine'}, {'data': 0}, **kwargs):
        data[e['ex_id']] = e.get(prop, 'none')
    json.dump(data, open(save_name, 'w'), indent=4, sort_keys=True)
'''

def export_index_files(props=['age', 'source-camera'], purpose='N2_aging',
                       store_blobs=True, out_dir=OUTDIR, tag='', **kwargs):
    """
    writes a json with a dictionary of ex_ids and the value of a correspoinding property
    :param prop: the property to be exported (key string for database documents)
    :param out_dir: the directory in which to save the json.
    """
    
    data = {}
    if store_blobs:
        id_type = 'blob_id'
    else:
        id_type = 'ex_id'

    for e in mr.mongo_query({'purpose': purpose, 'data_type': 'smoothed_spine'}, {'data': 0}, **kwargs):
        if e[id_type] not in data:
            data[e[id_type]] = {}
        for prop in props:
            data[e[id_type]][prop] = e.get(prop, 'none')            


    save_name = '{d}/index{tag}.json'.format(d=out_dir, tag=tag)
    json.dump(data, open(save_name, 'w'), indent=4, sort_keys=True)

    save_name = '{d}/index{tag}.txt'.format(d=out_dir, tag=tag)
    with open(save_name, 'w') as f:
        headers = [id_type] + props
        f.write(',\t'.join(headers))
        for dID in sorted(data):
            line = [dID] + [data[dID][i] for i in props]
            f.write(','.join(map(str, line)) + '\n')

if __name__ == '__main__':
    export_index_files(props=['duration'], tag='_duration')
    #export_ex_id_index(prop='age')
    #export_ex_id_index(prop='source-camera')
