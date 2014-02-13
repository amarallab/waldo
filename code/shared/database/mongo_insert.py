'''
Author: Peter Winter
Date: 
Description: This has many functions involved with importing data into the mongo database.
'''

# standard imports
import os
import sys
import pymongo
import numpy as np
import scipy.stats as stats
from itertools import izip

# path management
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
sys.path.append(project_directory)

# nonstandard imports
import mongo_support_functions as support_func
from mongo_retrieve import mongo_query
import manage_parts
from total_size import total_size
from settings.local import MONGO as MONGO
#from settings.data_settings import measurment_settings as 

default_mf = {'mean': np.mean,
              'std': np.std,
              '1st_q': lambda x: stats.scoreatpercentile(x, 25),
              'median': np.median,
              '3rd_q': lambda x: stats.scoreatpercentile(x, 75),
              # todo: autocorrelation
              }

# Initialize Globals
key_attributes = ['data_type', 'data', 'description', 'blob_id', 'part']
standard = [str, unicode]
numerical = [str, unicode, int, float]
toggle = [bool, int, float]
# the number 14680064 is hard coded as being smaller than the 16 MB mongo document limit
default_max_size = 14680064
#general = [None, str, unicode, int, float, list, dict, tuple]
entry_types = {'data_type': standard,
               'description': standard,
               'part': standard,
               'blob_id': standard,
               'ex_id': standard,
               'name': standard,
               'is_worm': toggle,
               'vid-flags': standard,
               'source-camera': standard,
               'purpose': standard,
               'strain': standard,
               'age': standard,
               'growth-medium': standard,
               'set-temp': standard,
               'stimulous': standard,
               'food': standard,
               'compounds': standard,
               'lid': standard,
               'vid-duration': numerical,
               'duration': numerical,
               'num-blobs-files': numerical,
               'num-images': numerical,
               'zoom': numerical,
               'pixels-per-mm': numerical, }

def times_to_timekeys(times):
    return [('%.3f' % float(t)).replace('.', '?') for t in times]

def times_to_timedict(times, data):
    timedict = {}
    for timekey, datum in izip(times_to_timekeys(times), data):
        timedict[timekey] = datum
    return timedict


def are_entry_types_valid(potential_entry, entry_types=entry_types, halt_import=True, verbose=True):
    '''
    Raises an error if you attempt to insert a doc into mongo that
    has keys you want with values that do not match your desired type.

    :param potential_entry: the dict that is about to be inserted into mongo
    :param entry_types: a dict containing potential attributes and acceptable types
                        for each of those attributes.
    '''
    for attribute in potential_entry:
        if str(attribute) in entry_types:
            acceptable_types = entry_types[str(attribute)]
            pot_attrb = potential_entry[attribute]
            if type(pot_attrb) != dict:
                error_string = 'Import Formatting Error:%s was type %s instead of types: %s\n%s' \
                               % (str(attribute),
                                  str(type(pot_attrb)),
                                  str(acceptable_types),
                                  str(pot_attrb))
            else:
                error_string = 'Import Formatting Error for data type:%s: %s was type %s instead of types: %s' \
                               % (str(potential_entry['data_type']),
                                  str(attribute),
                                  str(type(pot_attrb)),
                                  str(acceptable_types))

            if type(potential_entry[attribute]) not in acceptable_types:
                if halt_import:
                    assert False, error_string
                elif verbose:
                    print error_string
                    return False

    return True

def timedict_to_entry(timedict, other_entry, data_type, description):
    """
    Returns a entry (ie. Document) that follows the same template as other_entry but contains a new
    timedict, data_type, and description.

    :param timedict: timedict with desired data
    :param other_entry: database document in the desired format to use as a template. (most values will be copied)
    :param data_type: database key (string) describing data_type of new timedict
    :param description: a descriptive written string to inform a reader what data type is in this document'
    :return: new entry combining all inputs.
    """
    source_entry = other_entry.copy()
    if 'data' in source_entry:
        del source_entry['data']
    if 'part' in source_entry:
        del source_entry['part']

    new_entry = {}
    new_entry.update(source_entry)
    new_entry['description'] = description
    new_entry['data_type'] = data_type
    new_entry['data'] = timedict
    new_entry['part'] = '1of1'
    return new_entry


def insert_data_into_db(new_timedict, source_entry, data_type, description, insert_if_empty=False, **kwargs):
    '''
    creates a new database document for a timedict and inserts it into the database.

    :param new_timedict: the new data that is being added in timedict format
    :param source_entry: the entry from which the current data was calculated
                         (used to move all metadata + decision to split into parts or not)
    :param data_type: how the new data will be labeled in the database (str)
    :param description: how the new data will be described (str)
    :param insert_if_empty: toggle True/False if a document with an empty dictionary should be inserted.
    '''
    # for some data types, it is important to keep track of blank data, for others it indicates an error
    if insert_if_empty or len(new_timedict) > 0:
        new_entry = timedict_to_entry(new_timedict, source_entry, data_type, description)
        insert_blob_entries([new_entry], **kwargs)


def insert_blob_entries(new_entries, verbose=False, index=True, mongo_client=None, **kwargs):
    '''
    inserts a list of entries into the database with minimal processing. 
    If entry exists, it will be replaced.
    each new entry must contain the fields 'blob_id', 'data_type', 'data', 'part', and 'description'


    :param mongo_client: optional way to have mongo client passed in. if not, client is created on the fly.
    :param new_entries: list of entries (ie. documents) to be inserted into the database
    :param verbose: True/False toggle to display status messages when run.
    '''

    if mongo_client:
        close_mongo_when_done = False
        mongo_col = mongo_client[MONGO['database']][MONGO['blobs']]
    else:
        close_mongo_when_done = True
        print 'warning, creating new mongo_client'
        mongo_client, mongo_col = support_func.start_mongo_client(MONGO['ip'],
                                                                  MONGO['port'],
                                                                  MONGO['database'],
                                                                  MONGO['blobs'])
    for entry in new_entries:
        for key_attribute in key_attributes:
            err_msg = 'db insert error: {ka} is not in {dt} entry for {bi}'.format(ka=key_attribute,
                                                                                   dt=entry.get('data_type', '_'),
                                                                                   bi=entry.get('blob_id', '_'))
            assert key_attribute in entry, err_msg
        blob_id = entry['blob_id']
        data_type = entry['data_type']
        # The size of full data is checked and split into a list of however many entries required to stay below limits.
        entry_list = manage_parts.split_entry_into_part_entries(entry)
        if verbose:
            print 'inserting {dt} entry in {p} parts ({s} bytes)'.format(dt=data_type, p=len(entry_list),
                                                                         s=total_size(entry, verbose=False))
        # remove any entries from the list if they do not have proper types in each field.
        entry_list = [e for e in entry_list if are_entry_types_valid(entry, halt_import=False)]
        for i, entry in enumerate(entry_list):
            assert isinstance(blob_id, basestring)
            assert isinstance(data_type, basestring)
            assert isinstance(entry['part'], basestring)
            # if '_id' in the dict being inserted mongo will not update and won't tell you
            if '_id' in entry:
                del entry['_id']
            # Because 'part' sections may not line up perfectly, on import of first part remove previous data.
            if i == 0:
                if verbose:
                    print 'import step 1: removing previous documents for', blob_id, data_type
                mongo_col.remove({'blob_id': blob_id, 'data_type': data_type})
            if verbose:
                print i + 1, 'inserting', entry['part']
            mongo_col.update({'blob_id': blob_id, 'data_type': data_type, 'part': entry['part']},
                             entry, upsert=True)

    if index:
        index_blobs(mongo_col)

    if close_mongo_when_done:
        support_func.close_mongo_client(mongo_client)

def index_blobs(mongo_col=None):
    """ indexes certain fields are being indexed in the blobs collection.
    :param mongo_col: the pymongo collection object to be indexed.
    """
    if mongo_col:
        close_mongo_when_done = False
    else:
        close_mongo_when_done = True
        mongo_client, mongo_col = support_func.start_mongo_client(MONGO['ip'],
                                                                  MONGO['port'],
                                                                  MONGO['database'],
                                                                  MONGO['blobs'])


    # The main single field queries used throughout code
    mongo_col.ensure_index('ex_id')
    mongo_col.ensure_index('blob_id')
    mongo_col.ensure_index('purpose')
    mongo_col.ensure_index('data_type')

    # The most common double field queries.
    mongo_col.ensure_index([('ex_id', pymongo.ASCENDING), ('data_type', pymongo.ASCENDING)])
    mongo_col.ensure_index([('blob_id', pymongo.ASCENDING), ('data_type', pymongo.ASCENDING)])
    mongo_col.ensure_index([('purpose', pymongo.ASCENDING), ('data_type', pymongo.ASCENDING)])

    # other common indicies.
    mongo_col.ensure_index('blob_flag')
    mongo_col.ensure_index('strain')
    mongo_col.ensure_index('age')
    mongo_col.ensure_index('duration')
    mongo_col.ensure_index('food')
    mongo_col.ensure_index('part')
    mongo_col.ensure_index('compounds')
    mongo_col.ensure_index('pixels-per-mm')
    mongo_col.ensure_index('source-camera')

    if close_mongo_when_done:
        support_func.close_mongo_client(mongo_client)

def initialize_metric_document(id, time_range, col='worms', **kwargs):
    # pull current entry and add the new calculations
    """
    """
    assert col in ['worms', 'plates']
    if col == 'worms':
        id_type = 'blob_id'
    else:
        id_type = 'ex_id'

    results_entry = mongo_query({id_type: id, 'time_range': time_range}, find_one=True, col=col, **kwargs)
    # if none exists
    if results_entry == None:
        results_entry = mongo_query({id_type: id}, find_one=True, **kwargs)
        if 'part' in results_entry:
            del results_entry['part']
        if 'data_type' in results_entry:
            del results_entry['data_type']
        if id_type == 'ex_id' and 'blob_id' in results_entry:
            del results_entry['blob_id']
        results_entry['data'] = {}
        results_entry['time_range'] = time_range
        results_entry['flag'] = ''
    return results_entry

def insert_metric_documents(entry, col='worms', mongo_client=None):
    '''
    '''
    # check basic types match.
    are_entry_types_valid(entry, halt_import=True)
    assert col in ['worms', 'plates']
    if mongo_client:
        print 'using existing mongo client'
        close_mongo_when_done = False
        mongo_col = mongo_client[MONGO['database']][MONGO['blobs']]
    else:
        close_mongo_when_done = True
        mongo_client, mongo_col = support_func.start_mongo_client(MONGO['ip'],
                                                                  MONGO['port'],
                                                                  MONGO['database'],
                                                                  MONGO['blobs'])
    # depreciated
    '''
    mongo_conn, mongo_col = support_func.start_mongo_client(MONGO['ip'],
                                                            MONGO['port'],
                                                            MONGO['database'],
                                                            MONGO[col])
    '''
    # if '_id' in the dict being inserted mongo will not update and won't tell you
    if '_id' in entry:
        del entry['_id']

    err_msg = 'db insertion error: ex_id is not in {dt} entry for {bi}'.format(dt=entry.get('data_type', '_'),
                                                                               bi=entry.get('blob_id', '_'))
    assert 'ex_id' in entry, err_msg


    # worm and plate database documents have slightly different requirements.
    if col == 'worms':
        err_msg = 'db insertion error: blob_id is not in {dt} entry for {ei}'.format(dt=entry.get('data_type', '_'),
                                                                                     ei=entry.get('ex_id', '_'))
        assert 'blob_id' in entry, err_msg
        mongo_col.update({'blob_id': entry['blob_id']}, entry, upsert=True)
        mongo_col.ensure_index('blob_id')
    else:
        mongo_col.update({'ex_id': entry['ex_id']}, entry, upsert=True)

    mongo_col.ensure_index('ex_id')
    if close_mongo_when_done:
        support_func.close_mongo_client(mongo_client)

def insert_plate_document(document, mongo_client=None):
    if mongo_client:
        close_mongo_when_done = False
        mongo_col = mongo_client[MONGO['database']][MONGO['plates']]
    else:
        close_mongo_when_done = True
        mongo_client, mongo_col = support_func.start_mongo_client(MONGO['ip'],
                                                                  MONGO['port'],
                                                                  MONGO['database'],
                                                                  MONGO['plates'])
    if '_id' in document:
        del document['_id']

    ex_id = document['ex_id']
    data_type = document['type']
    mongo_col.update({'ex_id':ex_id, 'type':data_type}, document, upsert=True)

    mongo_col.ensure_index('ex_id')
    mongo_col.ensure_index('type')

    # The most common double field queries.
    mongo_col.ensure_index([('ex_id', pymongo.ASCENDING), ('data_type', pymongo.ASCENDING)])
    mongo_col.ensure_index([('purpose', pymongo.ASCENDING), ('data_type', pymongo.ASCENDING)])

    if close_mongo_when_done:
        support_func.close_mongo_client(mongo_client)


def compute_and_insert_measurements(dID, data, data_name, measure_functions=default_mf, time_range=(0, 1e20),
                                    col='worms', **kwargs):
    """
    :param data: list of floats
    :param data_name: data_type used to name stat in results entry
    :param measure_functions: dict of functions that are performed on data
    """
    # get existing entry or create a new one.
    results_entry = initialize_metric_document(dID, time_range, col=col, **kwargs)
    # compute all stats on this dataset and store in dict
    stat_dict = {}

    for mf in measure_functions:
        stat_dict[mf] = measure_functions[mf](data)

    for stat_type in stat_dict:
        stat_key = data_name + '_' + stat_type
        results_entry['data'][stat_key] = stat_dict[stat_type]
        # insert back into results dict.
    insert_metric_documents(results_entry, col=col, **kwargs)


def filter_skipped_and_out_of_range(data_timedict, time_range=(0, 1e20), verbose=False):

    t_data = sorted([(float(t.replace('?', '.')), data_timedict[t]) for t in data_timedict])
    # for 1d timeseries
    times, data = [], []
    datum_types = []
    failed_points = []
    for t, datum in t_data:
        if type(datum) not in datum_types:
            datum_types.append(type(datum))

        #print t, datum, type(datum)
        if (datum != 'skipped') and (time_range[0] <= t <= time_range[1]):
            times.append(t)
            data.append(datum)
        else:
            failed_points.append(datum)

        if verbose:
            print 'types of data in this dict are:', datum_types
            print len(data), 'data left'
    return times, data


if __name__ == '__main__':
    index_blobs()


