#!/usr/bin/env python
'''
Filename: file_manager.py

Description: holds many low-level scripts for finding, sorting, and saving files
in a rigid directory structure.
'''
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

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
import numpy as np

# nonstandard imports
from settings.local import LOGISTICS
from annotation.experiment_index import Experiment_Attribute_Index, organize_plate_metadata

INDEX_DIR = os.path.abspath(LOGISTICS['annotation'])
RESULT_DIR = os.path.abspath(LOGISTICS['results'])
EXPORT_PATH = os.path.abspath(LOGISTICS['export'])
WORM_DIR = os.path.abspath(LOGISTICS['worms'])
PLATE_DIR = os.path.abspath(LOGISTICS['plates'])
PREP_DIR = os.path.abspath(LOGISTICS['prep'])
DSET_DIR = os.path.abspath(LOGISTICS['dsets'])
NODENOTES_DIR = os.path.abspath(LOGISTICS['nodenotes'])
ANNOTATION_DIR = os.path.join(PREP_DIR, 'annotation')
TIME_SERIES_FILE_TYPE = LOGISTICS['time-series-file-type']

if TIME_SERIES_FILE_TYPE == 'hdf5':
    TIME_SERIES_FILE_TYPE = 'h5'
    # dont import if hdf5 not chosen. h5py may not be installed.
    from .h5_interface import write_h5_timeseries_base
    from .h5_interface import read_h5_timeseries_base

DSET_OPTIONS = ['d', 'ds', 'dset', 'dataset', 's', 'data_set']
RECORDING_OPTIONS = ['p', 'plate', 'ex_id', 'eid']
WORM_OPTIONS = ['w', 'worm', 'blob', 'b', 'bid', 'blob_id']

# def preprocessing_data(ex_id):
#     ''' returns a dict of preprocess data for an ex_id,
#     or an empty dict if nothing was found. '''
#     dset = get_dset(ex_id)
#     filename = 'threshold-{ds}.json'.format(ds=dset)
#     threshold_file = os.path.join(PREP_DIR, filename)
#     data = json.load(open(threshold_file)).get(ex_id, {})
#     return data

'''
class ColliderNodeNotes(object):
    def __init__(self, ex_id, directory=NODENOTES_DIR):
        f = '{eid}.csv'.format(eid=ex_id)
        self.eid =ex_id
        self.filedir = directory
        self.filename = os.join(directory, f)
        self.data = None

    def load(self):
        filename = self.filename
        err = '{eid} does not have file at: {p}'.format(eid=self.eid,
                                                        p=filename)
        assert os.path.isfile(filename), err
        data = pd.read_csv()
        self.data = data
        return data

    def dump(self, dataframe):
        pass

    def _return_set(self, dtype):
        if self.data is None:
            self.load()
        df = self.data[['bid', dtype]]
        bids = [b for (b, v) in df.Values() if v]
        return set(bids)

    def good(self):
        return self._return_set('good')

    def bad(self):
        return self._return_set('bad')

    def moved(self, thresh):
        if self.data is None:
            self.load()
        df = self.data[['bid', dtype]]
        bids = [b for (b, v) in df.Values() if v > thresh]
        return set(bids)
'''

class PrepData(object):
    def __init__(self, ex_id, prepdir=PREP_DIR):
        self.eid =ex_id
        self.filedir = os.path.join(prepdir, ex_id)
        self.files = []
        self.data_types = []
        self.refresh()

    def refresh(self):
        dir_is_there = os.path.exists(self.filedir)
        print(dir_is_there, 'is there')
        search = os.path.join(self.filedir, '*.csv')
        self.files = [f for f in iglob(search)]

        splitter = '{eid}-'.format(eid=self.eid)
        get_dt = lambda x: str(x).split(splitter)[-1].split('.csv')[0]
        self.data_types = [get_dt(f) for f in self.files]


    def load(self, data_type):
        assert data_type in self.data_types
        f = self.files[self.data_types.index(data_type)]
        return pd.read_csv(f)

    def dump(self, data_type, dataframe):
        filename = '{eid}-{dt}.csv'.format(eid=self.eid, dt=data_type)
        print(filename)
        dataframe.to_csv(os.path.join(self.filedir, filename))

    def good(self):
        """ returns a list containing only good nodes.

        returns
        -----
        good_list: (list)
            a list containing blob_ids
        """
        df = self.load('matches')[['bid', 'good']]
        return [b for (b, v) in df.Values() if v]


    def bad(self):
        """ returns a list containing only bad nodes.

        returns
        -----
        bad_list: (list)
            a list containing blob_ids
        """
        df = self.load('matches')[['bid', 'good']]
        return [b for (b, v) in df.Values() if not v]

    def moved(self, bl_threhold=2):
        pass


class Preprocess_File(object):
    # a class for interacting with preprocessing data
    # for an recording (ie. experiment id or ex_id)
    # or for a dataset (ie. dataset name or dset)

    # this gives access to region of interest data
    # (x, y, and radius) and threshold data

    def __init__(self, dset=None, ex_id=None):
        # specifiy either the experiment or the dataset.
        # consistancy checks.
        assert dset or ex_id, 'user must specify dset or ex_id'
        if dset and ex_id:
            err = 'dset does not match recorded dset for ex_id'
            assert dset == get_dset(ex_id), err
        if not dset:
            dset = get_dset(ex_id)

        self.path = ANNOTATION_DIR
        self.dset = dset
        self.ex_id = ex_id
        self.data = None

        #print(self.path)
        #print(self.dset)

        self.file = os.path.join(self.path,
                                 'threshold-{d}.json'.format(d=dset))

    def dump(self, data, ex_id=None):
        return json.dump(open(self.file, 'r'), data)

    def load(self):
        self.data = json.load(open(self.file, 'r'))
        return self.data

    def pull_data(self, ex_id):
        if not ex_id:
            ex_id = self.ex_id
        if not self.data:
            self.load()
        data = self.data[ex_id]
        return data

    def roi(self, ex_id=None):
        data = self.pull_data(ex_id)
        return {'x':data['x'], 'y':data['y'],
                'r':data['r']}

    def threshold(self, ex_id=None):
        data = self.pull_data(ex_id)
        return data['threshold']

def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def df_equal( df1, df2 ):
    """ Check if two DataFrames are equal, ignoring nans """
    return df1.fillna(1).sort(axis=1).eq(df2.fillna(1).sort(axis=1)).all().all()

def ensure_dir_exists(path):
    ''' recursivly creates path in filesystem, if it does not exist '''
    path = os.path.abspath(path)
    savedir = ''
    for i, d in enumerate(path.split('/')):
        if d:
            savedir += '/{d}'.format(d=d)
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
                print('created:{d}'.format(d=savedir))
    return savedir

class DataFile(object):
    #
    def __init__(self):
        pass


def get_dset(ex_id):
    ei = Experiment_Attribute_Index()
    return ei.attribute_index.get(ex_id, {}).get('dataset', 'something')
'''
def format_directory(ID_type, dataset='', tag='', ID='',
                    worm_dir=WORM_DIR, plate_dir=PLATE_DIR, dset_dir=DSET_DIR):

    if str(ID_type) in WORM_OPTIONS:
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
'''

def format_directory(ID_type, dataset='', tag='', ID=''):

    plate_or_dataset = list(RECORDING_OPTIONS).extend(DSET_OPTIONS)

    if str(ID_type) in WORM_OPTIONS:
        ex_id = '_'.join(ID.split('_')[:2])
        return '{path}/{eID}'.format(path=WORM_DIR.rstrip('/'), eID=ex_id)

    elif str(ID_type) in plate_or_dataset:
        # make sure we know which dset this plate belongs too

        if str(ID_type) in RECORDING_OPTIONS:
            path = PLATE_DIR
            if not dataset:
                dataset = get_dset(ex_id=ID)
        else:
            path = DSET_DIR

        return '{path}/{dset}/{tag}'.format(path=path,
                                            dset=dataset.rstrip('/'),
                                            tag=tag.rstrip('/'))


    assert False, 'ID_type not found. cannot use: {IDt}'.format(IDt=ID_type)


def get_good_blobs(ex_id, key='spine', worm_dir=WORM_DIR):
    search_dir = format_directory(ID=ex_id, ID_type='w')
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
        print('warning: ID_type is unknown')

    if dset == None:
        dset = 'unknown'
        print('warning: data set is unknown')

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
        file_dir = format_directory(ID_type, dataset=dset, tag=file_tag, ID=ID)

    ensure_dir_exists(file_dir)
    # Format the name of the file
    return  '{path}/{ID}-{dt}.{ft}'.format(path=file_dir, ID=ID,
                                           dt=data_type, ft=file_type)

def format_worm_filename(blob_id, data_type, file_type, worm_dir=WORM_DIR, ensure=False):
    ex_id = '_'.join(blob_id.split('_')[:2])
    file_dir = '{path}/{eID}'.format(path=worm_dir.rstrip('/'), eID=ex_id)
    if ensure:
        ensure_dir_exists(file_dir)
    filename =  '{path}/{ID}-{dt}.{ft}'.format(path=file_dir, ID=blob_id,
                                               dt=data_type, ft=file_type)
    return filename

def get_timeseries(ID, data_type, worm_dir=WORM_DIR):
    file_type=TIME_SERIES_FILE_TYPE
    filename = format_worm_filename(ID, data_type, file_type, worm_dir)
    if os.path.isfile(filename):
        # retrval method depends on file_type
        if file_type == 'json':
            data_dict = json.load(open(filename, 'r'))
            times, data = data_dict.get('time', []), data_dict.get('data', [])
        elif file_type == 'h5':
            times, data = read_h5_timeseries_base(filename)
        # print warning if file is empty
        if len(times)==0 and len(data)==0:
            print('No Time or Data Found! {dt} for {ID} not found'.format(dt=data_type, ID=ID))
        return times, data
    return None, None


def write_timeseries_file(ID, data_type, times, data, worm_dir=WORM_DIR):
    # if data not provided. write will fail. return False.
    if len(data) == 0:
        return False
    file_type = TIME_SERIES_FILE_TYPE
    filename = format_worm_filename(ID, data_type, file_type, worm_dir, ensure=True)
    # save method depends on file_type
    if file_type == 'json':
        json.dump({'time':times, 'data':data}, open(filename, 'w'))
    if file_type == 'h5':
        write_h5_timeseries_base(filename, times, data)
    return True


'''
class BaseTable(object):

    def __init__(self):
        self.dtype = ''
        self.ex_id = ''
        self.dset = ''
        self.basedir = ''
        self.ID_type = ''
        self.subdir = ''
        self.filedir = ''
        self.filename = ''
        self.filetype = ''

    def format_directory(self):
        base = self.basedir
        dset = self.dset
        if self.subdir:
            sd = self.subdir.rstrip('/')
            filedir =  '{base}/{dset}/{sd}'.format(base=base,
                                                   dset=dset,
                                                   sd=sd)
        else:
            sd = self.subdir.rstrip('/')
            filedir =  '{base}/{dset}'.format(base=base, dset=dset)
        return filedir

    def format_filename(self, ID):
        #errmsg = 'id must be string, not {i}'.format(i=ID)
        #assert isinstance(self.dset, basestring), errmsg
        path = self.filedir
        dt = self.dtype
        ft = self.filetype
        return '{path}/{ID}-{dt}.{ft}'.format(path=path, ID=ID,
                                              dt=dt, ft=ft)

    def dump(self, dataframe):
        ensure_dir_exists(self.file_dir)
        dataframe.to_hdf(self.filename, 'table', complib='zlib')

    def load(self):
        return pd.read_hdf(self.filename, 'table')


class DatasetTable(BaseTable):

    def __init__(self, dset, data_type, subdir=None):
        self.dtype = data_type
        self.ex_id = ''
        self.dset = dset
        self.basedir = DSET_DIR
        self.ID_type = 'dset'
        self.subdir = subdir
        self.filedir = self.format_directory()
        self.filename = self.format_filename(ID=dset)
        self.filetype = 'h5'


class PlateTable(BaseTable):

    def __init__(self, ex_id, data_type, subdir=None):
        self.dtype = data_type
        self.ex_id = ex_id
        self.dset = get_dset(ex_id)
        self.basedir = PLATE_DIR
        self.ID_type = 'plate'
        self.subdir = subdir
        self.filedir = self.format_directory()
        self.filename = self.format_filename(ID=ex_id)
        self.filetype = 'h5'


class BaseBlobDataFile(object):

    def __init__(self, bid, data_type):
        self.dtype = ''
        self.ex_id = ''
        self.dset = ''
        self.basedir = ''
        self.ID_type = ''
        self.subdir = ''
        self.filedir = ''
        self.filename = ''
        self.filetype = ''

    def format_directory(self):
        base = self.basedir
        dset = self.dset
        if self.subdir:
            sd = self.subdir.rstrip('/')
            filedir =  '{base}/{dset}/{sd}'.format(base=base,
                                                   dset=dset,
                                                   sd=sd)
        else:
            sd = self.subdir.rstrip('/')
            filedir =  '{base}/{dset}'.format(base=base, dset=dset)
        return filedir

    def format_filename(self, ID):
        #errmsg = 'id must be string, not {i}'.format(i=ID)
        #assert isinstance(self.dset, basestring), errmsg
        path = self.filedir
        dt = self.dtype
        ft = self.filetype
        return '{path}/{ID}-{dt}.{ft}'.format(path=path, ID=ID,
                                              dt=dt, ft=ft)

    def dump(self, dataframe):
        ensure_dir_exists(self.file_dir)
        dataframe.to_hdf(self.filename, 'table', complib='zlib')

    def load(self):
        return pd.read_hdf(self.filename, 'table')
'''


# TODO clean up read/write table
def write_table(ID, data_type, dataframe,
                ID_type='p',
                dset=None, file_tag='',
                file_dir=None, **kwargs):

    if dset==None and ID_type in ['p', 'plate']:
        dset=get_dset(ID)
    if dset==None and ID_type in ['s', 'dset',]:
        dset=get_dset(ID)

    # universal file formatting
    filename = format_filename(ID=ID,
                               ID_type=ID_type,
                               data_type=data_type,
                               file_type='h5',
                               dset=dset,
                               file_tag=file_tag,
                               file_dir=file_dir)

    dataframe.to_hdf(filename, 'table', complib='zlib')


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
    return pd.read_hdf(filename, 'table')


def write_metadata_file(ID, data_type, data, worm_dir=WORM_DIR):
    filename = format_worm_filename(ID, data_type, file_type='json', worm_dir=worm_dir, ensure=True)
    json.dump(data, open(filename, 'w'))

def get_metadata(ID, data_type='metadata', worm_dir=WORM_DIR):
    filename = format_worm_filename(ID, data_type, file_type='json', worm_dir=worm_dir)
    if os.path.isfile(filename):
        return json.load(open(filename, 'r'))
    return None

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
        print(tag)
        search_dir = os.path.abspath(format_directory(ID=ex_id, ID_type='plate',
                                                     dataset=dataset, tag=tag))
        search = '{d}/{eID}*'.format(d=search_dir, eID=ex_id)
        for rfile in iglob(search):
            print('removing:', rfile)
            os.remove(rfile)


def get_plate_files(dataset, data_type, tag='timeseries', path=None):
    if not path:
        path = format_directory(ID_type='plate', dataset=dataset, tag=tag)
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
    print(filename)
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

def get_annotations(dataset, data_type, label='all'):
    ex_ids, dfiles = get_plate_files(dataset=dataset, data_type=data_type)
    #print len(ex_ids), 'ex_ids found for', dataset, data_type
    #print len(dfiles)
    ids, days, files = [], [], []
    labels = []
    for eID, dfile in zip(ex_ids, dfiles):
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
