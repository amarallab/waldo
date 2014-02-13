#!/usr/bin/env python

'''
Filename: utilites.py
Description: 
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
from itertools import izip

# path definitions
SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(SHARED_DIR)

# nonstandard imports
from database.mongo_retrieve import pull_data_type_for_blob, mongo_query, timedict_to_list


def pull_datasets_for_data_type(query, data_type, min_timepoints=300):
    '''

    :param query:
    :param data_type:
    :param min_timepoints:
    query - dictionary for querying database (ex. {'strain':'N2'})
    data_type - data_type desired from database, find_ex_ids_to_update types stored in Settings
    min_timepoints - min number of value necessary to put blob data into datasets
    '''
    q = query.copy()
    q['data_type'] = data_type
    blobs_with_data = list(set([e['blob_id'] for e in mongo_query(q, {'blob_id': 1})]))
    print len(blobs_with_data), 'blobs have', data_type
    datasets = []
    all_data = []

    for blob_id in blobs_with_data[:]:
        #print blob_id
        blob_entry = pull_data_type_for_blob(blob_id, data_type)
        if len(blob_entry['data']) == 0:
            print 'warning : no data', blob_id, data_type
        else:
            times, data = timedict_to_list(blob_entry['data'])
            filtered_times, filtered_data = [], []
            if len(times) > 1:
                for t, d in zip(times, data):
                    if d != 'skipped':
                        filtered_times.append(t)
                        filtered_data.append(d)

            if len(filtered_data) > min_timepoints:
                datasets.append(filtered_data)
                for d in filtered_data: all_data.append(d)
    return datasets, all_data

def pull_data_from_results_db(query):
    """
    returns dicts of results and metadata with blob_ids as keys.
    :param query: typical mongo db query.
    """
    assert type(query) == dict
    result_dicts, metadata_dicts = {}, {}
    for entry in mongo_query(query, col='worm_collection'):
        blob_id = entry['blob_id']
        result_dicts[blob_id] = entry['data']
        assert isinstance(entry, dict)
        del entry['data']
        metadata_dicts[blob_id] = entry
    return result_dicts, metadata_dicts


def filter_results_single(data_type, results_dict):
    """
    returns a dictionary with the entries of the results_dict that
    contained the data_type string in the key

    results_dict is the 'data' part of the entries contained inside
    the results mongo db collection.

    :param data_type: one of the standard datatypes
    :param results_dict: a dictionary of
    :return: dictionary with only the entries
    """
    filtered_result_dict = {}
    for m_type in results_dict:
        if data_type == m_type[:len(data_type)]:
            filtered_result_dict[m_type] = results_dict[m_type]
    return filtered_result_dict

def filter_results_multi(data_type, results_dicts):
    """
    filters a dicionary of results dicts to remove data unrelated to
    data_type.

    :param data_type: type of data you wish to keep in the filtered results dict
    :param results_dicts: dict with blob_ids as keys and a results_dict as value
    :return: dict with blob_ids as keys and filtered_results_dict as value
    """
    filtered_results_dicts = {}
    for blob_id in sorted(results_dicts):
        filtered_results = filter_results_single(data_type, results_dicts[blob_id])
        filtered_results_dicts[blob_id] = filtered_results
    return filtered_results_dicts


def results_dict_to_lists(results_dicts, stat_type='mean'):
    blob_ids = []
    data = []
    for blob_id in results_dicts:
        result = None
        for stat in sorted(results_dicts[blob_id]):
            if stat_type == stat[-len(stat_type):]:
                result = results_dicts[blob_id][stat]
        if result != None:
            blob_ids.append(blob_id)
            data.append(result)
        else:
            print 'warning:', blob_id, 'does not have %s stats' %stat_type
    return blob_ids, data


def pair_metavalue_vs_datavalue(query, data_key, meta_key):
    '''
    return a list of tuples that contain values that correspond to
    one value stored in metadata and one in data of a list of database documents.
    The list of documents is generated either from a list of blob_ids
    or from a pymongo query.
    
    :param query: standard database query
    :param blob_ids: list of blob_id strings
    :param data_key:
    :param meta_key:
    '''
    assert isinstance(query, dict)
    assert isinstance(data_key, str)
    assert isinstance(meta_key, str)

    results_dicts, metadata_dicts = pull_data_from_results_db(query)
    paired_values = []
    for blob_id in results_dicts:
        data = results_dicts.get(blob_id, {})
        metadata = metadata_dicts.get(blob_id, {})
        #for data, metadata in izip(results_dicts, metadata_dicts):
        #print data, metadata
        #print data_key, data.get(data_key, 'none')
        #print meta_key, metadata.get(meta_key, 'none')
        if (data_key in data) and (meta_key in metadata):
            paired_values.append((metadata[meta_key], data[data_key]))
    return zip(*paired_values)
    
def get_multiple_matched_results(query=None, keys=None, blob_ids=None):
    '''
    return a list of tuples that contain values that correspond to each
    of the keys in a list of documents from the database. The list of documents
    is generated either from a list of blob_ids or from a pymongo query.
    
    :param query: standard database query
    :param blob_ids: list of blob_id strings
    :param keys: list of strings that will corespond to keys in the worms database
        
    example keys:
    speed_along_bl_mean
    speed_along_bl_median
    speed_perp_tail_bl_3rd_q
    '''
    assert type(keys) == list
    if query:
        assert not blob_ids
        assert type(query) == dict
    else:
        assert type(blob_ids) == list

    '''
    assert (query and not blob_ids) or (blob_ids and not query)
    def has_all_keys1(x):
        for k in keys:
            if k not in x:
                return False
        return True
    '''
    def has_all_keys(x):
        return len(keys) == len([k for k in keys if k in x])
    datasets = [[] for key in keys]
    results_dicts, metadata_dicts = pull_data_from_results_db(query)
    for blob_id in results_dicts:
        
        if blob_ids:
            blob_results = mongo_query({'blob_id': blob_id}, col='result_collection')[0]['data']
        else:    
            blob_results = results_dicts[blob_id]
        '''
        has_all_keys = True
        for key in keys:
            if key not in blob_results:
                has_all_keys = False

        a = has_all_keys
        b = has_all_keys1(blob_results)
        c = has_all_keys2(blob_results)
        if (not a) or (not b) or (not c):
            print 'test', a, b, c
        assert a == b
        assert a == c
        '''
        if has_all_keys(blob_results):
            for i, key in enumerate(keys):
                datasets[i].append(blob_results[key])
    return datasets

'''
def get_multiple_matched_results_for_ids(blob_ids, keys):

    #returns multiple matched

    datasets = [[] for _ in keys]
    for blob_id in blob_ids:
        blob_query = mongo_query({'blob_id': blob_id}, col='result_collection')[0]
        blob_results = blob_query['data']

        has_all_keys = True
        for key in keys:
            if key not in blob_results:
                has_all_keys = False

        if has_all_keys:
            for i, key in enumerate(keys):
                datasets[i].append(blob_results[key])

    return datasets
'''

if __name__ == '__main__':
    q = {'strain':'N2', 'age':'d3'}
    search_atrb = 'ex_id'
    print q, search_atrb
    entries = list(set([e[search_atrb] for e in mongo_query(q, {search_atrb:1}, col='result_collection')]))
    for e in entries: print e
