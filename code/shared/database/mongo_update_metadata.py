#!/usr/bin/env python

'''
Filename: mongo_update_metadata.py

Description: This file contains scripts involved with updating metadata fields in the mongo database in two ways.

update_ex_id_metadata -- this script rewrites metadata in the database to match the metadata in the index
spreadsheets.

mass_rename_fields -- this script is to clean up old documents after any renaming of fields in the database.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports

import os
import sys

# path definitions
project_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_dir)

# nonstandard imports
import settings.local as worm_env
import mongo_support_functions as support_func

def update_ex_id_metadata(ex_id, updated_attributes, col='blob_collection'):
    """
    updates all documents in the database for a given experiment to match the updated_attributes dict.

    :param ex_id: the recording id that should be updated.
    :param updated_attributes: a dictionary containing all attributes (key and value) that should be updated.
    :param col: specifies which collection to update.
    """
    assert isinstance(updated_attributes, dict)

    #print ex_id, updated_attributes


    mongo_client, mongo_col = support_func.start_mongo_client(worm_env.mongo_settings['mongo_ip'],
                                                                      worm_env.mongo_settings['mongo_port'],
                                                                      worm_env.mongo_settings['worm_db'],
                                                                      worm_env.mongo_settings[col])

    print '{ex_id} updating pixels-per-mm to {ppm}'.format(ex_id=ex_id,  ppm=updated_attributes['pixels-per-mm'])
    mongo_col.find({'ex_id': ex_id}, {'data': 0})
    mongo_col.update({'ex_id': ex_id}, {'$set': updated_attributes}, upsert=False, multi=True)
    support_func.close_mongo_client(mongo_client)


def mass_rename_fields():
    """
    This script goes through the database and renames several attributes.

    This script was written to fix issues involved with updating the database format and is only
    intended to be used on special occasions. modify the rename dict from within this script.
    """
    # Toggels
    rename_dict = {"ex-id": "ex_id",
                   "growth_medium": "growth-medium",
                   "lid?": "lid",
                   "num_blobs_files": "num-blobs-files",
                   "num_images": "num-images",
                   "pixels_per_mm": "pixels-per-mm",
                   "set_temp": "set-temp",
                   "source_camera": "source-camera",
                   "stimulous": "stimulus",
                   "vid_duration": "vid-duration",
                   "vid_flags": "vid-flags"}

    mongo_client, mongo_col = support_func.start_mongo_client(worm_env.mongo_settings['mongo_ip'],
                                                                      worm_env.mongo_settings['mongo_port'],
                                                                      worm_env.mongo_settings['worm_db'],
                                                                      worm_env.mongo_settings['blob_collection'])
    mongo_col.update({}, {'$rename': rename_dict}, upsert=False, multi=True)

    support_func.close_mongo_client(mongo_client)



if __name__ == '__main__':
    # toggle this to true if you want to rename fields in database.
    if False:
        mass_rename_fields()
