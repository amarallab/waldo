#!/usr/bin/env python
'''
Author: Peter
Description: tests if google_spreadsheet_interface behaving properly
'''
# standard imports
import sys
import os
import unittest
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# path definitinos
CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(SHARED_DIR)
sys.path.append(CODE_DIR)

# nonstandard imports
from settings.local import SPREADSHEET
from google_spreadsheet_interface import Spreadsheet_Interface

# Globals
USER = SPREADSHEET['user']
PASSWORD = SPREADSHEET['password']
ANNOTATION_SHEET = SPREADSHEET['spreadsheet']
SCALEING_SHEET = SPREADSHEET['scaling-factors']
ROW_ID = SPREADSHEET['row-id']
HEADERS = SPREADSHEET['columns']

#print USER
#print PASSWORD

'''
DEFAULT_DATA_DIR = LOGISTICS['filesystem_data']
DEFAULT_LOGISTICS_DIR = LOGISTICS['inventory']
DEFAULT_SAVE_DIR = LOGISTICS['annotation']
'''

class TestSpreadsheetInterface(unittest.TestCase):

    def _make_test_sheet(self, headers, row_names):
        ''' rows - list of dicts containing all headers except row_name
        upload_rows - list of dicts containing all headers including row_name
        '''
        #print headers, headers[1:]
        rows, upload_rows = [], []
        for i, r in enumerate(row_names):
            rows.append({})
            upload_rows.append({headers[0]: r})            
            for h in headers[1:]:
                #c = '{r}-{h}-{i}'.format(r=r, h=h, i=random.randint(1,10))
                c = random.randint(1,10)
                rows[i][h] = str(c)
                upload_rows[i][h] = str(c)
                
        #for r1, r2 in zip(rows, upload_rows):
        #    print r1, '\t', r2
        
        return rows, upload_rows

    def _assert_rows_equal(self, rows1, rows2):
        for r1, r2 in zip(rows1, rows2):
            self.assertEqual(r1.keys(), r2.keys())
                    
    def test_write_sheet_reproducabililty(self):
        # make spreadsheet connection
        spreadsheet = 'TestSheet'
        worksheet = 'Basic-Read-Write'
        si = Spreadsheet_Interface(email=USER, password=PASSWORD)
        
        # create fake spreadsheet contents
        headers = ('ex-id', 'head-1')
        row_names = ['1', '2', '123']
        rows, upload_rows = self._make_test_sheet(headers, row_names)
                
        # upload to                           
        si.upload_sheet(headers, upload_rows, spreadsheet, worksheet)
        h1, rn1, r1 = si.download_sheet(spreadsheet, worksheet)
        
        self.assertEqual(headers, h1)
        self.assertEqual(row_names, rn1)        
        self._assert_rows_equal(rows, r1)

    def test_header_suitability(self):
        # make spreadsheet connection
        spreadsheet = 'TestSheet'
        worksheet = 'chosen headers'        
        si = Spreadsheet_Interface(email=USER, password=PASSWORD)
        
        # create fake spreadsheet contents
        headers = HEADERS
        row_names = ['1']
        rows, upload_rows = self._make_test_sheet(headers, row_names)
                
        # upload to                           
        si.upload_sheet(headers, upload_rows, spreadsheet, worksheet)
        h1, rn1, r1 = si.download_sheet(spreadsheet, worksheet)
        
        self.assertEqual(tuple(headers), h1)
        self.assertEqual(row_names, rn1)        
        self._assert_rows_equal(rows, r1)
        
                
    
if __name__ == '__main__':
    unittest.main()
    
    si = Spreadsheet_Interface(email=USER, password=PASSWORD)
    a = si.download_sheet(spreadsheet='TestSheet', worksheet='Sheet1')
    headers, row_names, rows = a
    #print type(headers), headers
    #print type(row_names), row_names
    print type(rows)
    for i in rows:
        print i
        
    
