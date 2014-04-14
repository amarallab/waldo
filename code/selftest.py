#!/usr/bin/env python

'''
Filename: test_pipeline.py
Discription: 
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys

# path definitions
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
CODE_DIR = PROJECT_DIR + 'code/'
SHARED_DIR = CODE_DIR + 'shared/'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from annotation.experiment_index import Experiment_Attribute_Index
from settings.local import MONGO as mongo_settings
from tests.create_testfiles import write_index_file
#from waldo import import_ex_id
#from waldo import process_ex_id
from importing.process_spines import process_ex_id, just_process_centroid
# turn all this into unit tests.

TEST_DATA_DIR = CODE_DIR + 'tests/data/'
print os.path.abspath(TEST_DATA_DIR)
assert os.path.isdir(TEST_DATA_DIR)

def run_everything(**kwargs):

    ''' imports and processes test data. '''
    # get all ex_ids denoted as 'test'
    ei = Experiment_Attribute_Index()
    target_ex_ids = set(ei.return_ex_ids_with_attribute(key_attribute='dataset', attribute_value='selftest'))
    print '{N} test ids found: {l}'.format(N=len(target_ex_ids), l=target_ex_ids)
    if len(target_ex_ids) ==0:
        write_index_file()
        target_ex_ids = ['00000000_000001']
    # process all ex_ids
    for ex_id in sorted(target_ex_ids):
        print 'testing', ex_id
        '''
        process_ex_id(ex_id, 
                      # overwride 
                      min_body_lengths=0,
                      min_duration=1,
                      min_size=1,
                      data_dir=TEST_DATA_DIR,
                      debug=False)
        '''
        just_process_centroid(ex_id, 
                              # overwride 
                              min_body_lengths=0,
                              min_duration=1,
                              min_size=1,
                              data_dir=TEST_DATA_DIR,
                              debug=False)

if __name__ == '__main__':
    run_everything()

    '''
    try:
        mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                                   mongo_settings['worm_db'], mongo_settings['blob_collection'])
        run_everything(mongo_client=mongo_client)
    #check_if_test_worms_missing_fields()
    finally:
        mongo.close_mongo_client(mongo_client)
        #mongo_client.close()
    '''
