'''
functions for grabbing statistics from the data and comparing them.
'''


import os
import sys
code_directory =  os.path.dirname(os.path.realpath(__file__)) + '/../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)
from Shared.Code.Database.mongo_retrieve import mongo_query

def stat_by_blob_id(data_type, stat_type, blob_filter={}, **kwargs):
    '''
    returns a dictionary with keys:blob_id values: a float with one value

    inputs:
    data_type - key for f
    stat_type - mean, median, std, 1st, 3rd
    blob_filter - a dictionary that is added into the query 
    '''
    query = {'data_type':data_type}
    query.update(blob_filter)
    projection = {'blob_id':1, 'data':1}
    entries = mongo_query(query, projection, **kwargs)

    dataset_by_blob_id = {}
    for entry in entries:
        blob_id = entry['blob_id']
        blob_stat = entry['data'][stat_type] 
        assert blob_id not in dataset_by_blob_id
        assert type(blob_stat) in [float, int]
        dataset_by_blob_id[blob_id] = blob_stat
    return dataset_by_blob_id

def value_by_blob_id(attribute_type, blob_filter={}, **kwargs):
    '''
    returns a dictionary with keys:blob_id values: a float with one value

    inputs:
    data_type - key for f
    blob_filter - a dictionary that is added into the query 
    '''
    query = {'data_type':'metadata'}
    query.update(blob_filter)
    projection = {'blob_id':1, attribute_type:1}
    entries = mongo_query(query, projection, **kwargs )
    
    dataset_by_blob_id = {}
    for entry in entries:
        if attribute_type in entry:
            blob_id = entry['blob_id']
            blob_value = entry[attribute_type]
            if attribute_type == 'age': blob_value = int(blob_value.strip('d'))
            assert blob_id not in dataset_by_blob_id
            assert type(blob_value) in [float, int]
            dataset_by_blob_id[blob_id] = blob_value
    return dataset_by_blob_id

def pair_values_by_blob_id(first_value_by_blob_id, second_value_by_blob_id):
    '''
    accepts two dictionaries with blob_ids as keys and
    returns a list of tuples with values from matching blob ids.

    a blob id is not found in both dictionaries, it is not returned.
    '''
    
    paired_values = []
    if len(first_value_by_blob_id) != len(second_value_by_blob_id):
        print 'waring: the same stats are not present for the same blob_ids'
        print 'first list has', len(first_value_by_blob_id), 'blob_ids'
        print 'second list has', len(second_value_by_blob_id), 'blob_ids'
    for blob_id in first_value_by_blob_id:
        if blob_id in second_value_by_blob_id:
            v1 = first_value_by_blob_id[blob_id]
            v2 = second_value_by_blob_id[blob_id]
            paired_values.append((v1, v2))
    return paired_values

def pull_paired_stats(statname1, statname2, blob_filter={}):
    """
    extracts two types of stats from the database and pairs them by blob_id

    :param data_type1: type of data that should be pulled. str or unicode. should start with 'stats_'
    :param data_type2: similar to data_type1
    :param stat_type1:
    :param stat_type2:
    :param blob_filter:
    inputs:
    datatype1 and 2 -
    stat_type1 and 2 - the type of statistic that should be used. mean, median, std, 1st, 3rd
    blob_filter - dictionary that is added to database query
    """

    dataset1_by_blob_id = stat_by_blob_id(data_type1, stat_type1, blob_filter)
    dataset2_by_blob_id = stat_by_blob_id(data_type2, stat_type2, blob_filter)
    paired_stats = pair_values_by_blob_id(dataset1_by_blob_id, dataset2_by_blob_id)
    return paired_stats
