'''
Author: Peter Winter
Date: Nov. 26, 2012
Description:

These scripts are for extracting data,
'''

# standard imports
import os
import sys

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
print project_directory
print
print
sys.path.append(project_directory)

# nonstandard imports
from settings.local import MONGO as mongo_settings
import manage_parts
import mongo_support_functions as support_func

def timedict_to_list(timedict, remove_skips=False):
    timeseries = []
    for time_key in timedict:
        t = float(time_key.replace('?','.'))
        datum = timedict[time_key]
        if remove_skips and str(datum) == 'skipped':
            continue
        timeseries.append((t, datum))
    times, data = zip(*sorted(timeseries))
    return times, data

def pull_data_type_for_blob(blob_id, data_type, **kwargs):
    query = {'blob_id': blob_id, 'data_type': data_type}
    data_entries = mongo_query(query, **kwargs)
    assert len(data_entries) > 0, 'Error: {dt} not found for {bi}'.format(dt=data_type, bi=blob_id)
    combined_data_entry = manage_parts.combine_part_entries_to_entry(data_entries)
    return combined_data_entry


def mongo_query(query, projection={}, find_one=False, col='blobs', mongo_client=None, timeout=False, **kwargs):
    ''' simpelest query call. mirrors pymongo syntax. returns results as list. '''
    assert type(query) == dict
    assert type(projection) == dict

    if mongo_client:
        close_mongo_when_done = False
        mongo_col = mongo_client[mongo_settings['database']][mongo_settings[col]]
    else:
        close_mongo_when_done = True
        print 'warning, creating new mongo_client'
        mongo_client, mongo_col = support_func.start_mongo_client(mongo_settings['ip'],
                                                                  mongo_settings['port'],
                                                                  mongo_settings['database'],
                                                                  mongo_settings[col])

    if not find_one:
        if len(projection) == 0:
            entries = mongo_col.find(query, timeout=timeout)
        else:
            entries = mongo_col.find(query, projection, timeout=timeout)
        results = [entry for entry in entries]
    else:
        if len(projection) == 0:
            results = mongo_col.find_one(query, timeout=timeout)
        else:
            results = mongo_col.find_one(query, projection, timeout=timeout)

    if close_mongo_when_done:
        support_func.close_mongo_client(mongo_client)
    return results

def unique_blob_ids_for_query(query, col='blob_collection', **kwargs):
    return list(set([e['blob_id'] for e in mongo_query(query, {'blob_id':1}, col=col, **kwargs)]))

def unique_results_for_query(query=None, result_feild=None, col='blob_collection', mongo_client=None):
    #assert type(query) == dict
    #assert type(projection) == dict

    if mongo_client:
        close_mongo_when_done = False
        mongo_col = mongo_client[mongo_settings['database']][mongo_settings[col]]
    else:
        close_mongo_when_done = True
        print 'warning, creating new mongo_client'
        mongo_client, mongo_col = support_func.start_mongo_client(mongo_settings['ip'],
                                                                  mongo_settings['port'],
                                                                  mongo_settings['database'],
                                                                  mongo_settings[col])


    pipeline = [{'$project': {'purpose':1, 'ex_id': 1}},
                {'$match': {'purpose': 'N2_aging'}},
                {'$group': {'_id': 'N2_aging', 'ex_ids': {'$addToSet':'$ex_id'}}}
                #{'$unwind' : "$ex_ids" }
                ]
    # working console pipeline: [{$match: {'purpose':'N2_aging'}}, {$group:{_id:'ex_id', 'hey':{$addToSet: '$ex_id'}}}]
    # {$project:{'purpose':1, 'ex_id': 1}}, {$match: {'purpose':'N2_aging'}}, {$group:{_id:'ex_id', 'hey':{$addToSet: '$ex_id'}}}
    print pipeline
    results = mongo_col.aggregate(pipeline=pipeline)
    for doc in results:
        print 'doc', doc
    print 'done'

    if close_mongo_when_done:
        support_func.close_mongo_client(mongo_client)
    return results


def mongo_update(query, update, multi=False, col='blobs', mongo_client=None):
    ''' simpelest update call. mirrors pymongo syntax. '''
    assert type(query) == dict
    assert type(update) == dict

    if mongo_client:
        close_mongo_when_done = False
        mongo_col = mongo_client[mongo_settings['database']][col]
    else:
        close_mongo_when_done = True
        print 'warning, creating new mongo_client'
        mongo_client, mongo_col = support_func.start_mongo_client(mongo_settings['ip'],
                                                                  mongo_settings['port'],
                                                                  mongo_settings['database'],
                                                                  mongo_settings[col])
    mongo_col.update(query, update, multi=multi)
    if close_mongo_when_done:
        support_func.close_mongo_client(mongo_client)

def make_query_list(field='age', value_list=[], general_query={}):
    """
    returns a list of queries (dicts) that all have different values for the same field.

    :param field: the query field for which all queries will have different values
    :param value_list: the list of values that makes each query distinct
    :param general_query: the dict of fields and values that is shared by all the queries.
    :return: a list of queries (list of dicts)
    """
    queries = [{field: value} for value in value_list]
    if field in general_query:
        del general_query[field]
    for q in queries:
        q.update(general_query)
    return queries

if __name__ == '__main__':
    unique_results_for_query()
