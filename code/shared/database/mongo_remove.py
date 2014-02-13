import os
import sys
from bson.objectid import ObjectId


# nonstandard imports
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
sys.path.append(project_directory)

from settings.local import MONGO as mongo_settings
import mongo_support_functions as support_func
from mongo_retrieve import mongo_query
from mongo_retrieve import unique_blob_ids_for_query
from mongo_insert_new_data import check_entry_types

def remove_these_queries(queries):
    mongo_client, mongo_col = support_func.start_mongo_client(mongo_settings['mongo_ip'],
                                                                      mongo_settings['mongo_port'],
                                                                      mongo_settings['worm_db'],
                                                                      mongo_settings['blob_collection'])
    for query in queries:
        print 'Removing:', query
        mongo_col.remove(query)

    support_func.close_mongo_client(mongo_client)

def remove_entries_with_innapropriate_data_types(query={'data_type': 'metadata'}):
    # find all blob ids that match query
    blob_ids = unique_blob_ids_for_query(query)
    print len(blob_ids), 'blob_ids found for query', query


    for blob_id in blob_ids[:]:
        # check that all entries for a blob_id match the tests in 'are_entry_types_valid'
        entries = mongo_query({'blob_id': blob_id}, {'data_type': 1})
        removal_queries = []
        for entry in entries:
            entry_ok = check_entry_types(entry, halt_import=False)
            if not entry_ok:
                removal_queries.append({'_id': ObjectId(entry['_id'])})

        # report status.
        if len(removal_queries) == 0: print blob_id, 'is ok'
        else: print 'for %s, removing %i of %i entries' % (str(blob_id), len(removal_queries), len(entries))

        # remove entries that failed test.
        mongo_client, mongo_col = support_func.start_mongo_client(mongo_settings['mongo_ip'],
                                                                    mongo_settings['mongo_port'],
                                                                    mongo_settings['worm_db'],
                                                                    mongo_settings['blob_collection'])
        for removal_query in removal_queries:
            mongo_col.remove(removal_query)
        support_func.close_mongo_client(mongo_client)


if __name__ == '__main__':
    queries = [{'ex_id': {'$lt': '20130318_105552'}}]
    remove_these_queries(queries)
    #remove_entries_with_innapropriate_data_types({'age':'A4'})
    #remove_entries_with_old_data_types()
