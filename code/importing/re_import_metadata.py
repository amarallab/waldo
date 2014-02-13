#!/usr/bin/env python

'''
Filename: re_import_metadata.py
Description: scripts for updating the metadata in the database such that it matches the
values in our index spreadsheets. Aso has script to print differences for one ex_id.

Examples for possible usages:
update_all(key='purpose', value='disease_models')
update_all()
update_ex_id(ex_id='20130325_112559')
update_list = create_update_list(key='purpose', value='disease_models')
show_differences_between_index_and_database(ex_id='20130325_112559')
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import math

# path definitions
project_directory =  os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(project_directory)

# nonstandard imports
from experiment_index import Experiment_Attribute_Index
from database.mongo_retrieve import mongo_query
from database.mongo_update_metadata import update_ex_id_metadata

def create_update_list(key='purpose', value='N2_aging', ei=None, acceptable_error=0.001, **kwargs):
    """ Creates list of ex_ids that have a matching attribute:value field and have a larger than acceptable_error
    between the pixel-per-mm value in the database vs in the index spreadsheet.

    TODO: generalize process so it isn't just looking at pixels-per-mm

    :param ei: an Experiment_Index object
    :param key: field of
    :param value:
    :param acceptable_error: how far off the pixels-per-mm has to be to consider updating that ex_id
    :return:
    """
    if not ei:
        ei = Experiment_Attribute_Index()

    ex_ids = ei.return_ex_ids_with_attribute(key, value)
    update_list = []

    for ex_id in ex_ids:
        indexed_attributes = ei.return_attributes_for_ex_id(ex_id)
        db_attributes = mongo_query({'ex_id': ex_id, 'data_type': 'metadata'}, find_one=True, **kwargs)
        if db_attributes:

            db_sf = float(db_attributes.get(u'pixels-per-mm', 1.0))
            i_sf = float(indexed_attributes[u'pixels-per-mm'])
            if math.fabs(db_sf - i_sf) > acceptable_error:
                update_list.append(ex_id)
            print ex_id, math.fabs(db_sf - i_sf)
            '''
            databased = db_attributes.get(u'strain', '')
            indexed = indexed_attributes.get(u'strain', '')
            if indexed != databased:
                print ex_id, indexed, databased
                update_list.append(ex_id)
            '''
    print len(ex_ids), 'total'
    print len(update_list), 'to update'
    return update_list

def show_differences_between_index_and_database(ex_id, ei=None, **kwargs):
    """
    Prints all the fields in the index spreadsheet along with the values in the database.

    :param ex_id: experiment id
    :param ei: an Experiment_Index object
    """
    if not ei:
        ei = Experiment_Attribute_Index()
    index_attributes = ei.return_attributes_for_ex_id(ex_id)
    db_attributes = mongo_query({'ex_id': ex_id, 'data_type': 'metadata'}, find_one=True, **kwargs)
    print len(index_attributes), 'attributes in the Index Spreadsheet'
    print len(db_attributes), 'attributes in DB'
    print
    print 'DB Value\tIndex Value\tAttribute Name'
    for a in index_attributes:
        print '({db})\t({ix})\t{a}'.format(a=a, db=db_attributes.get(a, None), ix=index_attributes.get(a, None))

def update_ex_id(ex_id, ei=None):
    """
    changes all documents from an ex_id in the database so that they have the same values as it does in
    the Experiment_Index data. Any fields not contained in the index spreadsheets are not updated.

    :param ex_id: the id of the recording whose metadata you wish to update.
    :param ei: an Experiment_Index object
    """
    # if
    if not ei:
        ei = Experiment_Attribute_Index()

    index_attributes = ei.return_attributes_for_ex_id(ex_id)
    if 'ex-id' in index_attributes:
        del index_attributes['ex-id']
    print index_attributes
    update_ex_id_metadata(ex_id, index_attributes)

def update_all(key, value):
    """ goes through every ex_id in the index spreadsheets and makes all database documents of that ex_id have matching
    metadata fields.
    """
    ei = Experiment_Attribute_Index()
    update_list = create_update_list(key=key, value=value, ei=ei)
    for ex_id in update_list:
        print 'updating', ex_id
        update_ex_id(ex_id, ei)

if __name__ == '__main__':

    # Here are three examples on how these scripts can be run.
    #update_all(key='purpose', value='N2_aging') #value='disease_models')
    #update_ex_id(ex_id='20130325_112559')
    #update_list = create_update_list(key='purpose', value='disease_models')
    #update_list = create_update_list(key='purpose', value='N2_aging')

    ex_id = '20130318_142613'
    show_differences_between_index_and_database(ex_id=ex_id)
    #update_ex_id(ex_id=ex_id)
