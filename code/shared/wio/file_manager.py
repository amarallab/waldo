
#!/usr/bin/env python

'''
Filename: file_manager.py

Description: holds many low-level scripts for finding, sorting, and saving files
in a rigid directory structure.
'''


__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

#standard imports
import os
import sys
import json
from glob import iglob
import datetime
import pandas as pd
from itertools import izip
import numpy as np

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

#for i in sys.path:
#    print i
# nonstandard imports
from settings.local import LOGISTICS
from annotation.experiment_index import Experiment_Attribute_Index, organize_plate_metadata
    
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

#ensure_dir_exists(WORM_DIR)
        
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
                                                dset=dataset.rstrip('/'),
                                                tag=tag.rstrip('/'))
    else:
        assert False, 'ID_type not found. cannot use: {IDt}'.format(IDt=ID_type) 

    return file_dir

def get_good_blobs(ex_id, key='spine', worm_dir=WORM_DIR):
    search_dir = format_dirctory(ID=ex_id, ID_type='w')
    search = '{path}/*{key}.*'.format(path=search_dir, key=key)
    blobs = []
    for sf in iglob(search):
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
        return  '{path}/{ID}-{dt}.{ft}'.format(path=file_dir,
                                               ID=ID,
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

# TODO clean up read/write table
def write_table(ID, data_type, dataframe, 
                ID_type='w',
                dset=None, file_tag='',
                file_dir=None, **kwargs):
    # universal file formatting
    filename = format_filename(ID=ID, 
                               ID_type=ID_type,
                               data_type=data_type, 
                               file_type='h5',
                               dset=dset, 
                               file_tag=file_tag,
                               file_dir=file_dir)    
    dataframe.to_hdf(filename, 'fixed', complib='zlib')

def read_table(ID, data_type, ID_type='w', file_type='h5',
                dset=None, file_tag='',
                file_dir=None):
    # universal file formatting
    filename = format_filename(ID=ID, 
                               ID_type=ID_type,
                               data_type=data_type, 
                               file_type=file_type,
                               dset=dset, 
                               file_tag=file_tag,
                               file_dir=file_dir)
    return pd.read_hdf(filename, 'fixed')

def get_timeseries(ID, data_type, 
                   ID_type='w', file_type=TIME_SERIES_FILE_TYPE, 
                   dset=None, file_tag='',
                   file_dir=None, **kwargs):                   
    # universal file formatting
    filename = format_filename(ID=ID, ID_type=ID_type, data_type=data_type,                               
                               dset=dset, file_tag=file_tag,
                               file_type=file_type, file_dir=file_dir)
    #print os.path.abspath(filename), os.path.isfile(filename)
    #print 'tag', file_tag
    
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

# def search_db_for_data(blob_id, data_type, **kwargs):
#     try:
#         db_doc = pull_data_type_for_blob(blob_id, data_type, **kwargs)
#         print 'warning: could not find json file for', blob_id, data_type
#         # either split data into times and data or leave alone
#         # Stop everything if no to temp file or database doc.
#         return db_doc
#     except Exception as e:
#         print '\nFailure! could not locate data'
#         print 'blob: {bID}\t type:{dt}\n'.format(bID=blob_id, dt=data_type)
#         print e
#         return None

def get_ex_ids_in_worms(directory=WORM_DIR):
    search = '{path}/*'.format(path=directory.rstrip('/'))
    ex_ids = []
    for g in iglob(search):
        if os.path.isdir(g):
            ex_id = g.split('/')[-1]
            ex_ids.append(ex_id)
    # temporary code
    ex_ids2 = [g.split('/')[-1] for g in iglob(search) if os.path.isdir(g)]
    for e1, e2 in zip(ex_ids, ex_ids2):
        assert e1 == e2, 'update get_ex_id code'
    return ex_ids

def remove_plate_files(ex_id, file_tags):
    dataset = get_dset(ex_id)

    for tag in file_tags:
        print tag
        search_dir = os.path.abspath(format_dirctory(ID=ex_id, ID_type='plate',
                                                     dataset=dataset, tag=tag))
        search = '{d}/{eID}*'.format(d=search_dir, eID=ex_id)
        for rfile in iglob(search):
            print 'removing:', rfile
            os.remove(rfile)


def get_plate_files(dataset, data_type, tag='timeseries', path=None):
    if not path:
        path = format_dirctory(ID_type='plate', dataset=dataset, tag=tag)
    search = '{path}/*'.format(path=path.rstrip('/'))
    ex_ids, file_paths = [], []
    for file_path in iglob(search):
        if os.path.isfile(file_path):
            ex_id = file_path.split('/')[-1].split('-' +data_type)[0]
            # one small check before storing
            if len(ex_id.split('_')) == 2:
                ex_ids.append(ex_id)
                file_paths.append(file_path)
    return ex_ids, file_paths

# def format_plate_summary_name(ex_id, sum_type, dataset, data_type, path):
#     filename = format_filename(ID=ex_id, ID_type='plate',
#                                file_tag=sum_type,
#                                dset=dataset,
#                                data_type=data_type,
#                                file_dir=path,
#                                file_type='json')
#     return filename

# def format_dset_summary_name(data_type, dataset, sum_type, ID= None, dset_dir=None):
#     if not ID:
#         ID = dataset
#     filename = format_filename(ID=ID, ID_type='dset',
#                                data_type=data_type,
#                                file_tag=sum_type,
#                                dset=dataset,
#                                file_dir=dset_dir,
#                                file_type='json')
#
#     return filename


def write_dset_summary(data, data_type, dataset, sum_type, ID=None, dset_dir=None):
    #filename = format_dset_summary_name(data_type, dataset, sum_type, ID, dset_dir)
    if not ID:
        ID = dataset
    filename = format_filename(ID=ID, ID_type='dset',
                               data_type=data_type,
                               file_tag=sum_type,
                               dset=dataset,
                               file_dir=dset_dir,
                               file_type='json')
    print filename
    json.dump(data, open(filename, 'w'))


def read_dset_summary(data_type, dataset, sum_type='basic', ID=None, dset_dir=None):
    #filename = format_dset_summary_name(data_type, dataset, sum_type, ID, dset_dir)
    filename = format_filename(ID=ID, ID_type='dset',
                               data_type=data_type,
                               file_tag=sum_type,
                               dset=dataset,
                               file_dir=dset_dir,
                               file_type='json')
    return json.load(open(filename, 'r'))


# def write_plate_summary(data, ex_id, sum_type, dataset, data_type, path=None):
#     #filename = format_plate_summary_name(ex_id, sum_type, dataset, data_type, path)
#     filename = format_filename(ID=ex_id, ID_type='plate',
#                                file_tag=sum_type,
#                                dset=dataset,
#                                data_type=data_type,
#                                file_dir=path,
#                                file_type='json')
#     json.dump(data, open(filename, 'w'))
#
#
# def read_plate_summary(sum_type, dataset, data_type, path=None):
#     filename = format_filename(ID=ex_id, ID_type='plate',
#                                file_tag=sum_type,
#                                dset=dataset,
#                                data_type=data_type,
#                                file_dir=path,
#                                file_type='json')
#     #filename = format_plate_summary_name(ex_id, sum_type, dataset, data_type, path)
#     if os.path.isfile(filename):
#         return json.load(open(filename, 'r'))
#     return None
#
# def read_plate_timeseries(ex_id, dataset, data_type, tag='timeseries'):
#     times, data = get_timeseries(ID=ex_id,
#                                  data_type=data_type,
#                                  ID_type='p',
#                                  dset=dataset,
#                                  file_tag=tag)
#     #print len(times), len(data)
#     return times, data

# pre pandas
# def return_flattened_plate_timeseries(ex_id, dataset, data_type):
#     """
#     """
#     #times, data = parse_plate_timeseries_txt_file(dfile)
#     times, data = read_plate_timeseries(ex_id, dataset, data_type, tag='timeseries')
#     #times, data = parse_plate_timeseries_txt_file(dfile)
#     flat_data = []
#     if not len(times):
#         return []
#     for i, t_bin in enumerate(data):
#         flat_data.extend(list(t_bin))
#     # take data out of bins and remove nan values
#     flat_data = np.array(flat_data)
#     N_w_nan = len(flat_data)
#     flat_data = flat_data[np.logical_not(np.isnan(flat_data))]
#     N_wo_nan = len(flat_data)
#     if N_wo_nan != N_wo_nan:
#         print '{N} nans removed'.format(N=N_w_nan-N_wo_nan)
#     return flat_data

def return_flattened_plate_timeseries(ex_id, dataset, data_type):
    """
    """
    df = read_table(ID=ex_id, data_type=data_type, ID_type='p',
                    file_type='h5', dset=dataset, file_tag='timeseries')
    print df
    s = df.unstack()
    print s
    '''
    #times, data = parse_plate_timeseries_txt_file(dfile)
    flat_data = []
    if not len(times):
        return []
    for i, t_bin in enumerate(data):
        flat_data.extend(list(t_bin))
    # take data out of bins and remove nan values
    flat_data = np.array(flat_data)
    N_w_nan = len(flat_data)
    flat_data = flat_data[np.logical_not(np.isnan(flat_data))]
    N_wo_nan = len(flat_data)
    if N_wo_nan != N_wo_nan:
        print '{N} nans removed'.format(N=N_w_nan-N_wo_nan)
    return flat_data
    '''


def get_annotations(dataset, data_type, label='all'):
    ex_ids, dfiles = get_plate_files(dataset=dataset, data_type=data_type)
    #print len(ex_ids), 'ex_ids found for', dataset, data_type
    #print len(dfiles)
    ids, days, files = [], [], []
    labels = []
    for eID, dfile in izip(ex_ids, dfiles):
        #print eID, label
        hours, elabel, sub_label, pID, day = organize_plate_metadata(eID)
        if elabel not in labels:
            labels.append(elabel)
        if label == 'all' or str(label) == str(elabel):
            #print elabel, eID, day
            ids.append(eID)
            files.append(dfile)
            days.append(day)
    #print labels
    return ids, days, files