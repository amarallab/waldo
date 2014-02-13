'''
File: mongo_support_functions.py
Author: Peter Winter
Description: 
scripts to start/stop mongo client
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys

# path definitions
PROJECT_DIR =  os.path.dirname(os.path.realpath(__file__)) + '/../../../'
CODE_DIR = PROJECT_DIR + 'code/'
sys.path.append(CODE_DIR)

# nonstandard imports
from settings.local import MONGO

# default globals
USER = MONGO['user']
PASSWORD = MONGO['password']

def start_mongo_client(ip, port, database_name, collection_name, user=USER, pw=PASSWORD):
    '''
    Starts a connection to a MongoDB databse.
    Inputs:
    ip, port, database_name, collection_name
    auth = [username, password] if authentication required. [] if none required.

    Returns as objects:
    connection, collection
    '''
    from pymongo import Connection as MongoClient
    #from pymongo import MongoClient
    #Make sure that the port is an integer
    connection_string = 'mongodb://{user}:{pw}@{host}:{port}'.format(user=user, pw=pw, host=ip, port=port)
    #print connection_string
    #mongo_client = MongoClient(ip, int(port))
    mongo_client = MongoClient(connection_string)
    collection = mongo_client[database_name][collection_name]
    return mongo_client, collection

def close_mongo_client(mongo_client):
    mongo_client.close()
