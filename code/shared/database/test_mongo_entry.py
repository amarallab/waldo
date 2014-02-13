'''
Author: Peter
Date: November 29, 2012
Description:
picks a random ex_id in the mongo database and checks
1. opens and closes a mongo connection
2. Tests that the metadata matches for all that ex_id's 'metadata' entries. 
3. test if the metadata entries match all other entries for a blob

TODO:
4. test if xy_raw and size_raw match what's in mongo
5. test if something from .dat was imported correctly
6. test if .spine or .outline was imported correctly
'''

# standard imports
import sys
import os
import unittest
import random

# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(SHARED_DIR)
sys.path.append(CODE_DIR)

from importing.experiment_index import Experiment_Attribute_Index
from mongo_retrieve import get_attribute_from_entries_with_type, mongo_query
import mongo_support_functions as support_func


def unique_blob_ids_with_data_type(data_type):
    blobs_with_filtered_xy = mongo_query({'data_type':data_type}, {'blob_id':1})
    unique_blob_ids = list(set([entry['blob_id'] for entry in blobs_with_filtered_xy]))
    return unique_blob_ids

def return_all_ex_ids(attribute_filter={}):
    ''' returns list of all ex_ids in database. allows for filtering.'''
    assert type(attribute_filter) == dict
    if 'data_type' not in attribute_filter: attribute_filter['data_type'] = 'metadata'
    blob_entries = mongo_query(attribute_filter, {'ex_id':1})
    ex_ids = list(set([entry['ex_id'] for entry in blob_entries]))
    return ex_ids

def return_all_blob_ids(attribute_filter={}):
    ''' returns list of all blob_ids in database. allows for filtering.'''
    all_blob_ids = unique_blob_ids_with_data_type('metadata')
    return all_blob_ids

def pick_random_ex_id(attribute_filter={}):
    ''' randomly returns an ex_id from the database. allows filtering.'''
    all_db_ex_ids = return_all_ex_ids(attribute_filter)
    rand_ex_id = random.choice(all_db_ex_ids)
    return rand_ex_id

def pick_random_blob_id(attribute_filter={}):
    ''' randomly returns a blob_id from the database. allows filtering.'''
    all_blob_ids = return_all_blob_ids(attribute_filter)
    rand_blob_id = random.choice(all_blob_ids)
    return rand_blob_id

def find_blob_ids_without_data_type(data_type):
    ''' returns list of blob_ids that do not have db entries of a given data_type.
    '''
    all_blob_ids = get_attribute_from_entries_with_type('blob_id', 'metadata')
    blob_ids_with_type = get_attribute_from_entries_with_type('blob_id', data_type)
    blob_ids_without_data_type = []
    for blob_id in all_blob_ids:
        if blob_id not in blob_ids_with_type:
            blob_ids_without_data_type.append(blob_id)
    return blob_ids_without_data_type

def unadded_ex_ids_between_dates(start_date, end_date):
    '''
    Warning, gets ids from file directories.
    they may not be indexed properly.
    '''
    ei = Experiment_Attribute_Index()

    unadded_ex_ids = []
    print start_date, end_date
    indexed_ex_ids = ei.return_ex_ids_within_dates(start_date, end_date)
    #for i in indexed_ex_ids: print i

    databased_ex_ids = return_all_ex_ids()
    for ex_id in indexed_ex_ids:
        if ex_id not in databased_ex_ids:
            unadded_ex_ids.append(ex_id)

    return unadded_ex_ids



if len(sys.argv) <= 1:
    ex_id = pick_random_ex_id()
    print 'testing random ex_id:', ex_id
else:
    ex_id = unicode(sys.argv[1])
    print 'testing ex_id:', ex_id

class TestDatabaseEntries(unittest.TestCase):

    def setUp(self):
        self.start_conn = support_func.start_mongo_client
        self.stop_conn = support_func.close_mongo_client
        self.ex_id = ex_id
        date, time_stamp = ex_id.split('_')
        blob_ids, repeat = mongo_retrieve.get_attribute_from_entries_with_type('blob_id', 'metadata', 
                                                                               {'date':date, 'time_stamp':time_stamp})
        print len(blob_ids), 'blob ids found'
        self.blob_ids = blob_ids
        
    def test_ex_id_metadata(self):
        '''
        checks metadata entries of ex_id and makes sure they all contain
        the same metadata as the Annotated Index File
        '''
        

        ei = Experiment_Attribute_Index()
        ex_attributes = ei.return_attributes_for_ex_id(self.ex_id)
        for blob_id in self.blob_ids:
            metadata_entries = mongo_retrieve.mongo_query({'blob_id':blob_id, 'data_type':'metadata'})
            self.assertEqual(len(metadata_entries), 1)
            for metadata_entry in metadata_entries:
                #print metadata_entry['blob_id']
                for attribute in ex_attributes:
                    #print metadata_entry[attribute]
                    self.assertEqual(metadata_entry[attribute], ex_attributes[attribute])
        print 'metadata entries match the Experiment_Attribute_Index'

    def test_if_other_data_types_match_metadata(self):
        '''
        checks all entries of ex_id and makes sure they all contain
        the same metadata as the Annotated Index File
        '''
        key_attributes = ['data_type', 'data', 'part', 'description']

        ei = Experiment_Attribute_Index()
        ex_attributes = ei.return_attributes_for_ex_id(self.ex_id)
        for blob_id in self.blob_ids:
            metadata_entries = mongo_retrieve.mongo_query({'blob_id':blob_id, 'data_type':'metadata'})
            self.assertEqual(len(metadata_entries), 1)
            metadata_attributes = metadata_entries[0]
            self.assertEqual(type(metadata_attributes), dict)

            # the key attributes should not match between entries, hence remove them from the list.
            for attribute in key_attributes:
                if attribute in metadata_attributes: del metadata_attributes[attribute]


            all_blob_id_entries = mongo_retrieve.mongo_query({'blob_id':blob_id})
            for entry in all_blob_id_entries:
                if '_id' in metadata_attributes: del metadata_attributes['_id']
                for attribute in metadata_attributes:
                    if attribute not in entry: print entry['data_type'], 'has no attribute', attribute
                    elif entry[attribute] != metadata_attributes[attribute]: print entry['data_type'], 'not matching', attribute
                    self.assertEqual(entry[attribute], metadata_attributes[attribute])
        print 'all data_type entries match the attributes of the metadata entry'

if __name__ == '__main__':
    # if an ex_id is contained in sys.argv it breaks unittest.find_ex_ids_to_update(), so I remove any ex_ids
    if len(sys.argv) > 1: del sys.argv[1:]
    unittest.main()
