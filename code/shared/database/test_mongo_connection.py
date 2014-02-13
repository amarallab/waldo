'''
Author: Peter
Date: November
Description:
tries to open and close a connection using basic functions
and the functions in mongo_support_functions.
'''

# standard imports
import sys
from pymongo import Connection
import unittest

# nonstandard imports
sys.path.append('../')
#import Settings.worm_environment as worm_env
import mongo_support_functions as support_func
from Settings.data_settings import mongo_settings as worm_env

class TestDatabaseConnection(unittest.TestCase):

    def setUp(self):
        self.ip = worm_env.settings['mongo_ip']
        self.port = worm_env.settings['mongo_port']
        self.db_name = worm_env.settings['worm_db']
        self.collection = worm_env.settings['blob_collection']

    def test_simple_connection(self):
        ''' opens and closes a connection to
        the database specified in the Settings file
        '''
        
        print 'ip:%s' %self.ip
        print 'port:%s' %self.port
        connection = Connection(self.ip, self.port)
        print 'connection open'
        connection.close()
        print 'connection closed'

    def test_connection_function(self):

        mongo_client, mongo_col = support_func.start_mongo_client(self.ip,
                                                                          self.port,
                                                                          self.db_name,
                                                                          self.collection)
        print 'connection open'
        support_func.close_mongo_client(mongo_client)
        print 'connection closed'


if __name__ == '__main__':
    unittest.main()
