#!/usr/bin/env python

'''
Filename: import_rawdata_into_db.py

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
from glob import glob
from itertools import izip
 
# path definitions
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
CODE_DIR = PROJECT_DIR + 'code/'
SHARED_DIR = CODE_DIR + 'shared/'
sys.path.append(PROJECT_DIR)
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
#from database import mongo_query
#from database.mongo_insert import insert_blob_entries
#from database.mongo_insert import timedict_to_entry
from settings.local import LOGISTICS, FILTER

from annotation.experiment_index import Experiment_Attribute_Index
from wio.blob_reader import Blob_Reader
from wio.file_manager import write_timeseries_file, write_metadata_file

DATA_DIR = LOGISTICS['filesystem_data']

def create_entries_from_blobs_files(ex_id, min_body_lengths, min_duration, min_size, 
                                    max_blob_files=10000,
                                    data_dir=DATA_DIR, store_tmp=True, **kwargs):
    ''' creates a list of database documents out of all worthy blobs for a particular recording.

    :param ex_id: the experiment index of the recording
    :param min_body_lengths: the minimum number of body lengths a 'worthy' blob must travel in order to be stored.
    :param min_duration: the minimum number of seconds a 'worthy' blob must be tracked in order to be stored.
    :param min_size: the minimum number of pixels a 'worthy' blob must contain. (median pixels across all times)
    :param max_blob_files: if experiment contains more than this number, it is skipped to avoid deadlock
    '''
    # check if inputs are correct types and data directory exists
    assert type(min_body_lengths) in [int, float]
    assert type(min_body_lengths) in [int, float]
    assert type(min_duration) in [int, float]
    assert type(min_size) in [int, float]
    assert len(ex_id.split('_')) == 2
    path = data_dir + ex_id
    assert os.path.isdir(path)


    blob_files = sorted(glob(path+'/*.blobs'))    
    assert len(blob_files) < max_blob_files, 'too many blob files. this video will take forever to analyze.'+str(len(blob_files))
    
    BR = Blob_Reader(path=path, min_body_lengths=min_body_lengths, min_duration=min_duration, min_size=min_size)    
    
    raw_blobs = BR.pull_worthy_blobs()
    print len(raw_blobs), 'blobs found worthy'
    
    metadata_docs = create_metadata_docs(ex_id=ex_id, raw_blobs=raw_blobs)    

    if store_tmp:
        for local_id, blob in raw_blobs.iteritems():
            metadata = metadata_docs[local_id]
            blob_id = metadata['blob_id']
            write_metadata_file(data=metadata, blob_id=blob_id, data_type='metadata')
            write_timeseries_file(blob_id=blob_id, data_type='xy_raw',
                                  times=blob['time'], data=blob['xy'])
            write_timeseries_file(blob_id=blob_id, data_type='encoded_outline',
                                  times=blob['time'], data=blob['outline'])
            write_timeseries_file(blob_id=blob_id, data_type='aspect_ratio',
                                  times=blob['time'], data=blob['aspect_ratio'])

    # return a list of blob_ids
    blob_ids = [m['blob_id'] for m in metadata_docs.values()]
    return blob_ids
'''
def convert_lists_to_timedicts(blob_lists, types_to_convert=['xy', 'size', 'aspect_ratio', 'outline']):

    this function became necessary when I switched blob reader to no longer return blobs as timedicts.

    assert 'time' in blob_lists
    times = blob_lists['time']
    timekeys = [('%.3f' % float(f)).replace('.', '?') for f in times]
    new_blob = {'attributes': blob_lists['attributes']}
    for dtype in types_to_convert:
        timedict = {}
        for tk, datum in izip(timekeys, blob_lists[dtype]):
            if datum not in [None, '']:
                timedict[tk] = datum
        new_blob[dtype] = timedict
    return new_blob
'''
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
'''
def write_raw_blobs_to_db(raw_blobs, metadata_docs, **kwargs):
    new_entries = []
    for local_id, blob_with_lists in raw_blobs.iteritems():        
        assert local_id in metadata_docs, '{id} not in metadata docs!'.format(id=local_id)
        metadata_entry = metadata_docs[local_id]
        blob = convert_lists_to_timedicts(blob_with_lists)

        # store metadata_entry
        new_entries.append(metadata_entry)

        # Add in the data entries for insertion
        assert type(blob['size']) == dict
        data_type = 'size_raw'
        description = 'raw size data from mwt blobs file.'
        new_entry = timedict_to_entry(blob['size'], metadata_entry, data_type, description)
        #new_entry['part'] = '1of1'
        new_entries.append(new_entry)

        assert type(blob['xy']) == dict
        data_type = 'xy_raw'
        description = 'raw position data from mwt blobs file.'
        new_entry = timedict_to_entry(blob['xy'], metadata_entry, data_type, description)
        #new_entry['part'] = '1of1'
        new_entries.append(new_entry)

        assert type(blob['outline']) == dict
        data_type = 'encoded_outline'
        description = 'encoded outline data from mwt blobs file.'
        new_entry = timedict_to_entry(blob['outline'], metadata_entry, data_type, description)
        #new_entry['part'] = '1of1'
        new_entries.append(new_entry)

        assert type(blob['aspect_ratio']) == dict
        data_type = 'aspect_ratio'
        description = 'raw aspect_ratio from mwt blobs file.'
        new_entry = timedict_to_entry(blob['aspect_ratio'], metadata_entry, data_type, description)
        #new_entry['part'] = '1of1'
        new_entries.append(new_entry)
    # after every blobs file is processed 
    insert_blob_entries(new_entries, verbose=False, **kwargs)
'''
def import_ex_id(ex_id, min_body_lengths=FILTER['min_body_lengths'],
                 min_duration=FILTER['min_duration'], min_size=FILTER['min_size'],
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
    min_body_lengths = FILTER['min_body_lengths']
    min_duration = FILTER['min_duration']
    min_size = FILTER['min_size']

    if len(sys.argv) < 2:
        print sys.argv[0], '[ex_ids]'
        exit()

    ex_ids = sys.argv[1:]
    for ex_id in ex_ids:
        new_db_entries = create_entries_from_blobs_files(ex_id, min_body_lengths, min_duration, min_size)
