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
import datetime
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
from annotation.experiment_index import Experiment_Attribute_Index
    
INDEX_DIR = LOGISTICS['annotation']
RESULT_DIR = os.path.abspath(LOGISTICS['results'])
EXPORT_PATH = LOGISTICS['export']
WORM_DIR = LOGISTICS['worms']
PLATE_DIR = LOGISTICS['plates']
DSET_DIR = LOGISTICS['dsets']


TIME_SERIES_FILE_TYPE = LOGISTICS['time-series-file-type']

if TIME_SERIES_FILE_TYPE == 'hdf5':
    TIME_SERIES_FILE_TYPE = 'h5'
    # dont import if hdf5 not chosen. h5py may not be installed.    
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


def get_dset(ex_id):
    ei = Experiment_Attribute_Index()
    return ei.attribute_index.get(ex_id, {}).get('dataset', 'something')


def format_dirctory(ID_type, dataset='', tag='', ID='',
                    worm_dir=WORM_DIR, plate_dir=PLATE_DIR, dset_dir=DSET_DIR):
    if str(ID_type) in ['worm', 'w']:
        ex_id = '_'.join(ID.split('_')[:2])
        file_dir = '{path}/{eID}'.format(path=worm_dir.rstrip('/'), eID=ex_id)

    elif str(ID_type) in ['plate', 'p']:
        # make sure we know which dset this plate belongs too
        if dataset == None:
            dataset = get_dset(ex_id=ID)
        file_dir = '{path}/{dset}/{tag}'.format(path=plate_dir.rstrip('/'),
                                                    dset=dataset.rstrip('/'),
                                                    tag=tag.rstrip('/'))

    elif str(ID_type) in['dataset', 'dset', 's']:
        file_dir = '{path}/{dset}/{tag}'.format(path=dset_dir.rstrip('/'),
                                                dset=ID.rstrip('/'),
                                                tag=tag.rstrip('/'))
    else:
        assert False, 'ID_type not found. cannot use: {IDt}'.format(IDt=ID_type) 

    return file_dir

def get_good_blobs(ex_id, key='spine', worm_dir=WORM_DIR):
    search_dir = format_dirctory(ID=ex_id, ID_type='w')
    search = '{path}/*{key}.*'.format(path=search_dir, key=key)
    spine_files = glob(search)
    blobs = []
    for sf in spine_files:
        blob_id = sf.split('/')[-1].split('-{key}'.format(key=key))[0]
        blobs.append(blob_id)
    return blobs

def format_results_filename(ID, result_type, tag=None,
                            dset=None, ID_type='dset',
                            date_stamp=None,
                            file_type='png',
                            file_dir = RESULT_DIR,
                            ensure=False):
    # get fields in order before file creation
    if date_stamp == None:
        date_stamp = datetime.date.today()

    # standardize ID_type to a few options: worm, plate, dset, unkown
    ID_type = str(ID_type)
    if ID_type in ['dataset', 'dset', 's']:
        ID_type = 'dset'
        if dset == None:
            dset = ID        
    elif ID_type in ['plate', 'plates', 'p']:
        ID_type = 'plate'
    elif ID_type in ['worm', 'worms', 'w']:
        ID_type = 'worm'
    else:
        ID_type = 'unknown'
        print 'warning: ID_type is unknown'

    if dset == None:
        dset = 'unknown'
        print 'warning: data set is unknown'

    if tag == None:
        tag = ''
    else:
        tag = '-{t}'.format(t=tag)


    p = map(str, [file_dir, dset, ID_type, result_type, date_stamp])
    p = [i.rstrip('/') for i in p]
    save_dir = '{path}/{dset}/{dtype}/{rt}-{date}/'.format(path=p[0], dset=p[1], dtype=p[2], rt=p[3], date=p[4]) 
    if ensure:
        ensure_dir_exists(save_dir)
    filename = '{path}{ID}{tag}.{ft}'.format(path=save_dir, ID=ID, tag=tag, ft=file_type)
    return filename

def format_filename(ID, ID_type='worm', data_type='cent_speed', 
                    file_type='json',
                    file_dir = None,
                    dset=None, file_tag='',
                    worm_dir=WORM_DIR, plate_dir=PLATE_DIR, dset_dir=DSET_DIR):

    errmsg = 'id must be string, not {i}'.format(i=ID)
    assert isinstance(ID, basestring), errmsg

    if not file_dir:
        file_dir = format_dirctory(ID_type, dataset=dset, tag=file_tag, ID=ID)

    ensure_dir_exists(file_dir)
    # Format the name of the file
    if str(ID_type) in ['worm', 'w']:
        return '{path}/{ID}-{dt}.{ft}'.format(path=file_dir, 
                                              ID=ID,
                                              dt=data_type, 
                                              ft=file_type)
    elif str(ID_type) in ['plate', 'p']:
        # make sure we know which dset this plate belongs too
        return '{path}/{ID}-{dt}.{ft}'.format(path=file_dir, 
                                              ID=ID,
                                              dt=data_type, 
                                              ft=file_type)
    elif str(ID_type) in['dset', 's']:
        return  '{path}/{tag}{dt}.{ft}'.format(path=file_dir,
                                               tag=file_tag,
                                               dt=data_type, 
                                               ft=file_type)
    
def write_timeseries_file(ID, data_type, times, data, 
                          ID_type='w', file_type=TIME_SERIES_FILE_TYPE,
                          dset=None, file_tag='',
                          file_dir=None, **kwargs):
    # if data not provided. write will fail. return False.
    if len(data) == 0:
        return False

    # universal file formatting
    filename = format_filename(ID=ID, 
                               ID_type=ID_type,
                               data_type=data_type, 
                               file_type=file_type,
                               dset=dset, 
                               file_tag=file_tag,
                               file_dir=file_dir)
    # save method depends on file_type
    if file_type == 'json':
        json.dump({'time':times, 'data':data}, open(filename, 'w'))
    if file_type == 'h5':        
        write_h5_timeseries_base(filename, times, data)
    return True

def get_timeseries(ID, data_type, 
                   ID_type='w', file_type=TIME_SERIES_FILE_TYPE, 
                   dset=None, file_tag='',
                   file_dir=None, **kwargs):                   
    # universal file formatting
    filename = format_filename(ID=ID, ID_type=ID_type, data_type=data_type,                               
                               dset=dset, file_tag=file_tag,
                               file_type=file_type, file_dir=file_dir)
    #print 'tag', file_tag
    #print filename, os.path.isfile(filename)
    if os.path.isfile(filename):
        # retrval method depends on file_type
        if file_type == 'json':
            data_dict = json.load(open(filename, 'r'))
            times, data = data_dict.get('time', []), data_dict.get('data', [])                    
        elif file_type == 'h5':        
            times, data = read_h5_timeseries_base(filename)
        # print warning if file is empty
        if len(times)==0 and len(data)==0:
            print 'No Time or Data Found! {dt} for {ID} not found'.format(dt=data_type, ID=ID)
        return times, data
    return None, None
    

def write_metadata_file(ID, data_type, data, ID_type='w', file_dir=None, **kwargs):
    # universal file formatting
    filename = format_filename(ID=ID, ID_type=ID_type, data_type=data_type,                                                              
                               file_type='json', file_dir=file_dir)                               
    json.dump(data, open(filename, 'w'))
        
def get_metadata(ID, data_type='metadata', ID_type='w', file_dir=None, **kwargs):
    filename = format_filename(ID=ID, ID_type=ID_type, data_type=data_type,
                               file_type='json', file_dir=file_dir)
    
    if os.path.isfile(filename):
        return json.load(open(filename, 'r'))
    return None


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

def get_ex_ids_in_worms(directory=WORM_DIR):
    search = '{path}/*'.format(path=directory.rstrip('/'))
    ex_ids = []
    for g in glob(search):
        if os.path.isdir(g):
            ex_id = g.split('/')[-1]
            ex_ids.append(ex_id)
    return ex_ids

def get_ex_ids(query, **kwargs):
    # return a list of unique ex_id names for a query
    return list(set([e['ex_id'] for e in mongo_query(query=query, projection={'ex_id':1}, **kwargs)]))

def get_blob_ids(query, **kwargs):
    # return a list of unique blob_id names for a query
    return list(set([e['blob_id'] for e in mongo_query(query=query, projection={'blob_id':1}, **kwargs)]))

'''
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


def write_metadata_file(blob_id, data_type, data, json_dir=WORM_DIR):
    json_file = format_json_filename(blob_id, data_type, json_dir=json_dir)
    json.dump(data, open(json_file, 'w'))



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

def clear_json_file(blob_id, data_type='all'):
    blob_path = '{path}/{bID}'.format(path=json_dir, bID=blob_id)
    json_file = '{path}/{dt}.json'.format(path=blob_path, dt=data_type)
    silent_remove(json_file)
    if data_type == 'all' or len(glob.glob(blob_path + '/*')) == 0:
        os.rmdir(blob_path)

def read_json_file(blob_id, data_type, json_dir=WORM_DIR):
    json_file = format_json_filename(blob_id, data_type, json_dir=json_dir)
    if os.path.isfile(json_file):
        return json.load(open(json_file, 'r'))    
    return None

'''    
        

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
       
ensure_dir_exists(WORM_DIR)
        
    
