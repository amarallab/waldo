#!/usr/bin/env python

'''
Filename: temp_cache.py

Description:
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

#standard imports
import os
import sys
import json
from glob import glob
#from itertools import izip
 
# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from database.mongo_retrieve import pull_data_type_for_blob, timedict_to_list, mongo_query
from database.mongo_insert import insert_data_into_db, times_to_timedict
from settings.local import LOGISTICS
from wio.blob_reader import Blob_Reader
    
INDEX_DIR = LOGISTICS['annotation']
EXPORT_PATH = LOGISTICS['export']

WORM_DIR = PROJECT_HOME + '/data/worms/json/'
PLATE_DIR = PROJECT_HOME + '/data/plates/'
DSET_DIR = PROJECT_HOME + '/data/dsets/'
H5_DIR = PROJECT_HOME + '/data/worms/h5/'

USE_JSON = True
if not USE_JSON:
    from h5_interface import write_h5_timeseries_base
    from h5_interface import read_h5_timeseries_base

        
def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
   
def ensure_dir_exists(path):
    ''' recursivly creates path in filesystem, if it does not exist '''
    path = os.path.abspath(path)
    savedir = ''
    for i, d in enumerate(path.split('/')):
        if d:
            savedir += '/{d}'.format(d=d)        
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
                print 'created:{d}'.format(d=savedir)
    return savedir

def manage_save_path(out_dir, path_tag, ID, data_type):
    ''' returns a unique descriptive file name to store data and makes sure path to it exists'''
    # get directories in order
    out_dir = '{d}/{tag}/'.format(d=out_dir.rstrip('/'), tag=path_tag.lstrip('/'))
    out_dir = ensure_dir_exists(out_dir.rstrip('/'))

    #now_string = time.ctime().replace('  ', '_').replace(' ', '_')
    #now_string = now_string.replace(':', '.').strip()
    save_name = '{path}/{ID}-{dt}'.format(path=out_dir, ID=ID, dt=data_type)
    #print save_name
    return save_name

def get_ex_ids(query, **kwargs):
    ''' return a list of unique ex_id names for a query'''
    return list(set([e['ex_id'] for e in mongo_query(query=query, projection={'ex_id':1}, **kwargs)]))

def get_blob_ids(query, **kwargs):
    ''' return a list of unique blob_id names for a query'''
    return list(set([e['blob_id'] for e in mongo_query(query=query, projection={'blob_id':1}, **kwargs)]))

def get_ex_id_metadata(ex_id, json_dir=JSON_DIR):
    search_path = '{path}/{eID}/*metadata.json'.format(path=json_dir.rstrip('/'), 
                                                       eID=ex_id.rstrip('/'))
    #print search_path
    json_files = glob(search_path)
    #print json_files
    return json.load(open(json_files[0], 'r'))
    
def format_json_filename(blob_id, data_type, json_dir):
    errmsg = 'blob_id must be string, not {i}'.format(i=blob_id)
    assert isinstance(blob_id, basestring), errmsg
    ex_id = '_'.join(blob_id.split('_')[:2])
    blob_path = '{path}/{eID}'.format(path=json_dir, eID=ex_id)
    ensure_dir_exists(blob_path)
    json_file = '{path}/{bID}-{dt}.json'.format(path=blob_path, bID=blob_id,
                                               dt=data_type)
    return json_file

def format_h5_path(blob_id, data_type, h5_dir):
    errmsg = 'blob_id must be string, not {i}'.format(i=blob_id)
    assert isinstance(blob_id, basestring), errmsg    
    ex_id = '_'.join(blob_id.split('_')[:2])
    h5_dir = h5_dir.rstrip('/')
    file_path = '{path}/{eID}'.format(path=h5_dir, eID=ex_id)
    ensure_dir_exists(file_path)
    h5_file = '{path}.h5'.format(path=file_path)
    h5_dataset = '{bID}/{dt}'.format(bID=blob_id, dt=data_type)
    return h5_file, h5_dataset    
    #h5_file = '{path}/{bID}-{dt}.json'.format(path=file_path, bID=blob_id,
    #                                          dt=data_type)
    #return h5_file #, h5_dataset

#def write_outlines(blob_id, data_type, times, data, h5_dir=H5_DIR):
#    h5_file = format_h5_path(blob_id, data_type, h5_dir)
#    write_h5_outlines(h5_file, times, data)
    
def write_h5_timeseries(blob_id, data_type, times, data, h5_dir=H5_DIR):
    h5_file = format_h5_path(blob_id, data_type, h5_dir)
    write_h5_timeseries_base(h5_file, times, data)
    
def read_h5_timeseries(blob_id, data_type, h5_dir=H5_DIR):
    h5_file = format_h5_path(blob_id, data_type, h5_dir)
    times, data = read_h5_timeseries_base(h5_file)
    #print data
    return times, data

def write_timeseries_file(blob_id, data_type, times, data, json_dir=JSON_DIR):
    '''
    types = []
    for t in data:
        try:
            for i in t:
                if type(i) not in types:
                    types.append(type(i))

        except:
            pass
        if type(t) not in types:
            types.append(type(t))
    print types                                        
    print data_type
    '''
    json_file = format_json_filename(blob_id, data_type, json_dir=json_dir)
    json.dump({'time':times, 'data':data}, open(json_file, 'w'))
    #write_h5_timeseries(blob_id, data_type, times, data)    
        
def write_metadata_file(blob_id, data_type, data, json_dir=JSON_DIR):
    json_file = format_json_filename(blob_id, data_type, json_dir=json_dir)
    json.dump(data, open(json_file, 'w'))

def read_json_file(blob_id, data_type, json_dir=JSON_DIR):
    json_file = format_json_filename(blob_id, data_type, json_dir=json_dir)
    if os.path.isfile(json_file):
        return json.load(open(json_file, 'r'))    
    return None

def clear_json_file(blob_id, data_type='all'):
    blob_path = '{path}/{bID}'.format(path=json_dir, bID=blob_id)
    json_file = '{path}/{dt}.json'.format(path=blob_path, dt=data_type)
    silent_remove(json_file)
    if data_type == 'all' or len(glob.glob(blob_path + '/*')) == 0:
        os.rmdir(blob_path)
                                  
def get_timeseries(blob_id, data_type, search_db=True, **kwargs):
    #return read_h5_timeseries(blob_id, data_type)
    # default: look in json file and split into 'time' and 'data'
    data_dict = read_json_file(blob_id=blob_id, data_type=data_type)
    # temp file not located, and search_db=True, attempt to find data in database.
    if search_db and not isinstance(data_dict, dict):
        data_dict = search_db_for_data(blob_id, data_type=data_type, **kwargs)

    # if data source found
    if isinstance(data_dict, dict):
        times, data = data_dict.get('time', []), data_dict.get('data', [])        

        if not times and not data:
            print 'No Time or Data Found! {dt} for {bi} not found'.format(dt=data_type, bi=blob_id)
        return times, data
    # if data source not found
    return None, None
        
def get_metadata(blob_id, data_type='metadata', search_db=True, **kwargs):
    
    metadata = read_json_file(blob_id=blob_id, data_type=data_type)
    # default: look in json file and split into 'time' and 'data'
    found_it = False
    # look in temp file but do not split results
    if metadata != None:
        return metadata
    # temp file not located, and search_db=True, attempt to find data in database.
    elif search_db:
        metadata = search_db_for_data(blob_id, data_type='metadata')
    return metadata
    
def search_db_for_data(blob_id, data_type, **kwargs):
    try:
        db_doc = pull_data_type_for_blob(blob_id, data_type, **kwargs)
        print 'warning: could not find json file for', blob_id, data_type
        # either split data into times and data or leave alone
        # Stop everything if no to temp file or database doc.
        return db_doc
    except Exception as e:
        print '\nFailure! could not locate data'
        print 'blob: {bID}\t type:{dt}\n'.format(bID=blob_id, dt=data_type)
        print e
        return None

'''                             
def store_data_in_db(blob_id, data_type, data, description, db_doc=None, **kwargs):
    if not db_doc:
        db_doc = read_json_file(blob_id=blob_id, data_type='metadata')
    if not db_doc:
        try:
            db_doc = mongo_query({'blob_id':blob_id, 'data_type':'metadata'}, find_one=True)
            print 'warning: could not find json file for', blob_id, 'metadata'
        except Exception as e:
            print '\nFailure! could not locate data'
            print 'blob: {bID}\t type:{dt}\n'.format(bID=blob_id, dt='metadata')
            print e
            assert False
    insert_data_into_db(data, db_doc, data_type=data_type, 
                        description=description, **kwargs)                        
    return db_doc



def store_data_in_db(blob_id, data_type, times, data, description, db_doc=None, **kwargs):
    # convert back to timedict form. then insert
    timedict = times_to_timedict(times, data)
    store_timedict_in_db(blob_id, data_type, timedict, description, db_doc, **kwargs)
'''    
       
ensure_dir_exists(JSON_DIR)
        
    
