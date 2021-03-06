#!/usr/bin/env python

'''
Filename: initialize_recording.py

Description: sifts through raw data files created by Multi-Worm Tracker ('*.blobs' files).
Blobs that meet minimal requirements have their data reformatted and are inserted into the database along with some
metadata recorded in the experiemnt_attribute_index.py
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

#standard imports
import os
import sys
from glob import iglob
from itertools import izip
import numpy as np
import numbers
import shutil

# path definitions
PROJECT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
CODE_DIR = os.path.join(PROJECT_DIR, 'code')
SHARED_DIR = os.path.join(CODE_DIR, 'shared')
JOINING_DIR = os.path.join(SHARED_DIR, 'joining')
sys.path.append(PROJECT_DIR)
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)
sys.path.append(JOINING_DIR)

# nonstandard imports
from conf import settings
from annotation.experiment_index import Experiment_Attribute_Index
from wio.blob_reader import Blob_Reader
from wio.file_manager import write_timeseries_file, write_metadata_file, format_directory

from tapeworm import Taper


def use_tapeworm():
    return (settings.JOINING['method'] == 'tapeworm') # turned on and off in settings file.

def create_entries_from_blobs_files(ex_id, min_body_lengths, min_duration, min_size,
                                    max_blob_files=10000, overwrite = True,
                                    data_dir=None, store_tmp=True, **kwargs):

    if data_dir is None:
        data_dir = settings.MWT_DATA_ROOT
    # remove existing directory if overwite == true
    init_dir = os.path.abspath(format_directory(ID=ex_id,ID_type='worm'))
    print 'overwite previous data is: {o}'.format(o=overwrite)
    if overwrite and os.path.isdir(init_dir):
        print 'removing: {d}'.format(d=init_dir)
        shutil.rmtree(init_dir)

    # initialize the directory
    if use_tapeworm():
        return tape_worm_creation(ex_id, min_body_lengths, min_duration, min_size,
                                  max_blob_files, overwrite=overwrite,
                                  data_dir=data_dir, store_tmp=True)
    else:
        return blob_reader_creation(ex_id, min_body_lengths, min_duration, min_size,
                                    max_blob_files, overwrite=overwrite,
                                    data_dir=data_dir,store_tmp=True)

def midline_lengths(midlines):
    ''' accepts a list of midlines (a list of xy tuples) and returns the length of all lines
    that are not None or have enough xy coords.
    '''
    lengths = []
    # the midline portion starts with a '%' and ends either with '%%' or line end.
    for line in midlines:
        if line == None:
            continue
        if 2 > len(line) > 4:
            continue
        try:
            X, Y = zip(*line)
            dx = np.diff(np.array([float(x) for x in X]))
            dy = np.diff(np.array([float(y) for y in Y]))
            lengths.append(np.sqrt(dx**2 +dy**2))
        except Exception as e:
            print e
    return lengths

def tape_worm_creation(ex_id, min_body_lengths, min_duration, min_size,
                         max_blob_files=10000,
                         data_dir=None, store_tmp=True, overwrite=False, **kwargs):
    ''' creates a list of database documents out of all worthy blobs for a particular recording.

    :param ex_id: the experiment index of the recording
    :param min_body_lengths: the minimum number of body lengths a 'worthy' blob must travel in order to be stored.
    :param min_duration: the minimum number of seconds a 'worthy' blob must be tracked in order to be stored.
    :param min_size: the minimum number of pixels a 'worthy' blob must contain. (median pixels across all times)
    :param max_blob_files: if experiment contains more than this number, it is skipped to avoid deadlock
    '''
    if data_dir is None:
        data_dir = settings.MWT_DATA_ROOT
    # check if inputs are correct types and data directory exists
    for arg in (min_body_lengths, min_duration, min_size):
        assert isinstance(arg, numbers.Number)
    assert len(ex_id.split('_')) == 2
    path = data_dir + ex_id
    assert os.path.isdir(path), 'Error. path not found: {path}'.format(path=path)

    blob_files = sorted(iglob(os.path.join(path, '*.blobs')))
    assert len(blob_files) < max_blob_files, 'too many blob files. this video will take forever to analyze.'+str(len(blob_files))

    # get indexed data about this recording.
    ei = Experiment_Attribute_Index()
    ex_attributes = ei.return_attributes_for_ex_id(ex_id)
    if ex_attributes == None:
        print 'ex id not found by Experiment_Attribute_Index'
        exit()


    plate = Taper(directory=path, min_move=min_body_lengths,
               min_time=min_duration, verbosity=1)
    plate.load_data()


    blob_ids = []

    for local_id, blob in plate.segments():

        # go through blobs and convert them into
        unique_blob_id = '{eID}_{local}'.format(eID=ex_id, local=local_id)
        blob_ids.append(unique_blob_id)

        #print local_id, blob.keys()
        if store_tmp:
            # create full metadata entry for this blob to later use in the creation of data entries
            metadata_entry = {'blob_id': unique_blob_id.strip(),
                              'local_blob_id': local_id.strip(),
                              'ex_id': ex_id.strip(),
                              'is_worm': 0,
                              'data_type': 'metadata',
                              'data': None,
                              #'part': '1of1',
                              'description': 'metadata for blob without any additional data',
                              # get all blob components
                              'local_segments': blob.get('segments', [local_id]),
                              #TODO: add calculated attributes
                              'start_time': float(blob['time'][0]),
                              'stop_time': float(blob['time'][-1]),
                              'duration': float(blob['time'][-1]) - float(blob['time'][0]),

                              'bl_dist':0,
                              'midline_median': np.median(midline_lengths(blob['midline'])),
                              }

            # add descriptive attribues from the experiment to metadata
            for atrib, value in ex_attributes.iteritems():
                metadata_entry[atrib.strip()] = value

            # write the metadata entry
            write_metadata_file(data=metadata_entry, ID=unique_blob_id, data_type='metadata')
            # write centriod positions
            write_timeseries_file(ID=unique_blob_id, data_type='xy_raw',
                                  times=blob['time'], data=blob['centroid'])

            def map_string(a_list):
                return [str(i) if i else '' for i in a_list]

            # write encoded outlines
            x, y = zip(*blob['contour_start'])
            x, y, l, o = [map_string(i) for i in (x, y, blob['contour_encode_len'], blob['contour_encoded'])]
            outlines = zip(x, y, l, o)
            write_timeseries_file(ID=unique_blob_id, data_type='encoded_outline',
                                  times=blob['time'], data=outlines)


            # TODO: add aspect ratio
            # write aspect ratio
            #write_timeseries_file(ID=unique_blob_id, data_type='aspect_ratio',
            #                      times=blob['time'], data=blob['aspect_ratio'])

    return blob_ids


def blob_reader_creation(ex_id, min_body_lengths, min_duration, min_size,
                         max_blob_files=10000,
                         data_dir=None, store_tmp=True, overwrite=False, **kwargs):
    ''' creates a list of database documents out of all worthy blobs for a particular recording.

    :param ex_id: the experiment index of the recording
    :param min_body_lengths: the minimum number of body lengths a 'worthy' blob must travel in order to be stored.
    :param min_duration: the minimum number of seconds a 'worthy' blob must be tracked in order to be stored.
    :param min_size: the minimum number of pixels a 'worthy' blob must contain. (median pixels across all times)
    :param max_blob_files: if experiment contains more than this number, it is skipped to avoid deadlock
    '''
    if data_dir is None:
        data_dir = settings.MWT_DATA_ROOT
    # check if inputs are correct types and data directory exists
    assert type(min_body_lengths) in [int, float]
    assert type(min_body_lengths) in [int, float]
    assert type(min_duration) in [int, float]
    assert type(min_size) in [int, float]
    assert len(ex_id.split('_')) == 2
    path = data_dir + ex_id
    assert os.path.isdir(path), 'Error. path not found: {path}'.format(path=path)


    blob_files = sorted(iglob(path+'/*.blobs'))
    assert len(blob_files) < max_blob_files, 'too many blob files. this video will take forever to analyze.'+str(len(blob_files))

    BR = Blob_Reader(path=path, min_body_lengths=min_body_lengths,
                     min_duration=min_duration, min_size=min_size)

    raw_blobs = BR.pull_worthy_blobs()
    print len(raw_blobs), 'blobs found worthy'

    metadata_docs = create_metadata_docs(ex_id=ex_id, raw_blobs=raw_blobs)



    if store_tmp:
        for local_id, blob in raw_blobs.iteritems():
            metadata = metadata_docs[local_id]
            blob_id = metadata['blob_id']
            #print blob_id
            write_metadata_file(data=metadata, ID=blob_id, data_type='metadata')
            write_timeseries_file(ID=blob_id, data_type='xy_raw',
                                  times=blob['time'], data=blob['xy'])

            #print len(blob['outline'])
            #outlines = np.zeros(shape=(len(blob['outline']), 4), dtype=str)
            #for i, o in enumerate(blob['outline']):
            #    #print i, o
            #    if len(o) == 4:
            #        outlines[i] = np.array(o, dtype=str)
            outlines = np.array(blob['outline'], ndmin=2, dtype='str')
            write_timeseries_file(ID=blob_id, data_type='encoded_outline',
                                  times=blob['time'], data=outlines)
            #write_timeseries_file(ID=blob_id, data_type='encoded_outline',
            #                      times=blob['time'], data=blob['outline'])
            write_timeseries_file(ID=blob_id, data_type='aspect_ratio',
                                  times=blob['time'], data=blob['aspect_ratio'])

    # return a list of blob_ids
    blob_ids = [m['blob_id'] for m in metadata_docs.values()]
    return blob_ids

def reformat_outline(outlines):
    # turns outlines in point format with variable length
    # into two matricies, one with x values, one with y values
    #
    xs, ys, N = [], [], []
    for o in outlines:
        x, y = zip(*o)
        xs.append(x)
        ys.append(y)
        N.append(len(x))

    N_max = max(N)

    ox = np.zeros(shape=[len(N), N_max])
    oy = np.zeros(shape=[len(N), N_max])
    for i, (x, y) in enumerate(izip(xs, ys)):
        ox[i][:len(x)] = np.array(x)
        oy[i][:len(x)] = np.array(y)
    return ox, oy

def create_metadata_docs(ex_id, raw_blobs):

    # read experiment attributes from experiment attribute index
    ei = Experiment_Attribute_Index()
    ex_attributes = ei.return_attributes_for_ex_id(ex_id)
    if ex_attributes == None:
        print 'ex id not found by Experiment_Attribute_Index'
        exit()

    metadata_docs = {}
    # go through blobs and convert them into
    for local_id in raw_blobs:
        blob = raw_blobs[local_id]
        unique_blob_id = ex_id + '_' + local_id

        # create full metadata entry for this blob to later use in the creation of data entries
        metadata_entry = {'blob_id': unique_blob_id.strip(),
                          'local_blob_id': local_id.strip(),
                          'ex_id': ex_id.strip(),
                          'is_worm': 0,
                          'data_type': 'metadata',
                          'data': None,
                          'part': '1of1',
                          'description': 'metadata for blob without any additional data'}

        # add descriptive attribues from the experiment to metadata
        for k in ex_attributes:
            metadata_entry[k.strip()] = ex_attributes[k]
        # add aggregate attributes specific to blob to metadata
        for k in blob['attributes']:
            metadata_entry[k.strip()] = blob['attributes'][k]
        metadata_docs[local_id] = metadata_entry
    return metadata_docs

def import_ex_id(ex_id, min_body_lengths=settings.FILTER['min_body_lengths'],
                 min_duration=settings.FILTER['min_duration'], min_size=settings.FILTER['min_size'],
                 overwrite=True, **kwargs):
    ''' Imports raw data from one experiment into the database.

    :param ex_id: experiment id string
    :param min_body_lengths: the minimum number of body lengths a 'worthy' blob must travel in order to be stored.
    :param min_duration: the minimum number of seconds a 'worthy' blob must be tracked in order to be stored.
    :param min_size: the minimum number of pixels a 'worthy' blob must contain. (median pixels across all times)
    :param overwrite: if entries exist in
    '''
    # check if entries for ex_id already exist, if not, read blobs files
    import_ex_id = True
    if not overwrite:
        entries = mongo_query({'ex_id': ex_id, 'data_type': 'metadata'}, {'blob_id': 1, 'duration': 1}, **kwargs)
        if len(entries) > 0:
            import_ex_id = False

    if import_ex_id:
        create_entries_from_blobs_files(ex_id, min_body_lengths, min_duration, min_size, **kwargs)

if __name__ == '__main__':
    # toggle constraints
    min_body_lengths = settings.FILTER['min_body_lengths']
    min_duration = settings.FILTER['min_duration']
    min_size = settings.FILTER['min_size']

    if len(sys.argv) < 2:
        print sys.argv[0], '[ex_ids]'
        exit()

    ex_ids = sys.argv[1:]
    for ex_id in ex_ids:
        new_db_entries = create_entries_from_blobs_files(ex_id, min_body_lengths, min_duration, min_size)
