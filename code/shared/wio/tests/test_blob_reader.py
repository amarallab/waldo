#!/usr/bin/env python

'''
Filename: test_blob_reader.py

Description: unit tests to test blob_reader.py
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import unittest
import math
from Import.Code import blob_reader

class TestBlobReader(unittest.TestCase):
    def setUp(self):
        self.acceptable_error = 10e-6

    def test_parse_outline(self):
        # if outline in line, there should be one '%%' denoting outline start.
        # no '%%'
        fakeline = '% hey % hey % hey %'
        outline = blob_reader.parse_outline_from_line(fakeline)
        self.assertEqual(outline, None)
        # too many '%%'
        fakeline = '%% hey %%'
        outline = blob_reader.parse_outline_from_line(fakeline)
        self.assertEqual(outline, None)
        # after '%%' there should be 4 cols.
        # too few cols after '%%'
        fakeline = 'hey % hey %% 1 2 3'
        outline = blob_reader.parse_outline_from_line(fakeline)
        self.assertEqual(outline, None)
        # too many cols after '%%'
        fakeline = 'hey % hey %% 1 2 3 4 5'
        outline = blob_reader.parse_outline_from_line(fakeline)
        self.assertEqual(outline, None)
        # wrong types passed in
        fakeline = 'hey % hey %% a 2 3 4 5'
        outline = blob_reader.parse_outline_from_line(fakeline)
        self.assertEqual(outline, None)
        # wrong types passed in
        fakeline = 'hey % hey %% 1 2.2 3 4 5'
        outline = blob_reader.parse_outline_from_line(fakeline)
        self.assertEqual(outline, None)
        # wrong types passed in
        fakeline = 'hey % hey %% 1 2 3, 4 5'
        outline = blob_reader.parse_outline_from_line(fakeline)
        self.assertEqual(outline, None)
        # correct 1
        fakeline = 'hey % hey %% 1 2 3 abc'
        outline = blob_reader.parse_outline_from_line(fakeline)
        self.assertEqual(outline, ((1, 2), 3, 'abc'))

    def test_parse_midline(self):
        acceptable_error = self.acceptable_error
        # if spine in line, there should be one '%' denoting outline start.
        # using diagnonal line as fakedata
        fakeline = 'hey %'
        for i in range(11): fakeline += ' ' + str(i) + ' ' + str(i)
        midline = blob_reader.calculate_midline_from_line(fakeline)
        self.assertTrue(abs(midline - math.sqrt(2 * (10 ** 2))) < acceptable_error)
        # longer diagnonal line, including outline portion in line after a '%%'
        fakeline = 'hey %'
        for i in range(11): fakeline += ' ' + str(i * 2) + ' ' + str(i * 2)
        fakeline += ' %% outline'
        midline = blob_reader.calculate_midline_from_line(fakeline)
        self.assertTrue(abs(midline - math.sqrt(2 * (20 ** 2))) < acceptable_error)
        # same diagnonal line, too many '%'
        fakeline = 'hey %'
        for i in range(11): fakeline += ' ' + str(i * 2) + ' ' + str(i * 2)
        fakeline += ' %%% outline'
        midline = blob_reader.calculate_midline_from_line(fakeline)
        self.assertEqual(midline, None)
        # same diagnonal line, too few coordinates
        fakeline = 'hey %'
        for i in range(10): fakeline += ' ' + str(i * 2) + ' ' + str(i * 2)
        fakeline += ' %% outline'
        midline = blob_reader.calculate_midline_from_line(fakeline)
        self.assertEqual(midline, None)
        # straight vertical line
        fakeline = 'hey %'
        for i in range(11): fakeline += ' ' + str(i) + ' ' + str(0)
        fakeline += ' %% outline'
        midline = blob_reader.calculate_midline_from_line(fakeline)
        self.assertTrue(abs(midline - 10) < acceptable_error)
        # straight horizontal line
        fakeline = 'hey %'
        for i in range(11): fakeline += ' ' + str(10) + ' ' + str(i)
        fakeline += ' %% outline'
        midline = blob_reader.calculate_midline_from_line(fakeline)
        self.assertTrue(abs(midline - 10) < acceptable_error)
        # horizontal, too many coordinates
        fakeline = 'hey %'
        for i in range(12): fakeline += ' ' + str(10) + ' ' + str(i)
        fakeline += ' %% outline'
        midline = blob_reader.calculate_midline_from_line(fakeline)
        self.assertEqual(midline, None)
        # horizontal, wrong type in columns
        fakeline = 'hey % a a'
        for i in range(10): fakeline += ' ' + str(10) + ' ' + str(i)
        fakeline += ' %% outline'
        midline = blob_reader.calculate_midline_from_line(fakeline)
        self.assertEqual(midline, None)

    def test_is_blob_worty(self):
        test_prop = {'size_median': 10, 'midline_median': 5, 'start_time': 0,
                     'stop_time': 100, 'duration': 100, 'bl_dist': 3}
        # start with criterion low enough for blob to pass
        min_bl, min_dur, min_size = 2.99, 99.9, 9.9
        isworty = blob_reader.is_blob_worthy(test_prop, min_bl, min_dur, min_size)
        self.assertEqual(isworty, True)
        # bl too small
        min_bl, min_dur, min_size = 3.01, 99.9, 9.9
        isworty = blob_reader.is_blob_worthy(test_prop, min_bl, min_dur, min_size)
        self.assertEqual(isworty, False)
        # duration too small
        min_bl, min_dur, min_size = 2.99, 100.1, 9.9
        isworty = blob_reader.is_blob_worthy(test_prop, min_bl, min_dur, min_size)
        self.assertEqual(isworty, False)
        # size too small
        min_bl, min_dur, min_size = 2.99, 99.9, 10.1
        isworty = blob_reader.is_blob_worthy(test_prop, min_bl, min_dur, min_size)
        self.assertEqual(isworty, False)
        # test_prop is missing bl_dist
        test_prop = {'size_median': 10, 'midline_median': 5, 'start_time': 0,
                     'stop_time': 100, 'duration': 100}
        min_bl, min_dur, min_size = 2.99, 99.9, 9.9
        isworty = blob_reader.is_blob_worthy(test_prop, min_bl, min_dur, min_size)
        self.assertEqual(isworty, False)
        # test_prop is missing size_median
        test_prop = {'start_time': 0, 'stop_time': 100, 'duration': 100, 'bl_dist': 1}
        min_bl, min_dur, min_size = 2.99, 99.9, 9.9
        isworty = blob_reader.is_blob_worthy(test_prop, min_bl, min_dur, min_size)
        self.assertEqual(isworty, False)

    def test_compute_basic_properties(self):
        acceptable_error = self.acceptable_error
        # blob does not contain all needed attributes
        fakeblob = {'time_list': [i / 10.0 for i in range(101)]}
        propdict = blob_reader.compute_basic_properties(fakeblob)
        self.assertEqual(propdict, {})
        # correct calculations
        fakeblob = {'time_list': [i / 10.0 for i in range(101)],
                    'xy_list': [(i, i) for i in range(101)],
                    'size': {},
                    'midline': {}, }
        for i in range(101):
            fakeblob['size'][str(i)] = int(i)
            fakeblob['midline'][str(i)] = int(i)
        propdict = blob_reader.compute_basic_properties(fakeblob)
        self.assertEqual(propdict['size_median'], 50.0)
        self.assertEqual(propdict['midline_median'], 50.0)
        self.assertEqual(propdict['start_time'], 0)
        self.assertEqual(propdict['stop_time'], 10.0)
        self.assertEqual(propdict['duration'], 10.0)
        self.assertTrue(abs(propdict['bl_dist'] - (math.sqrt(2 * (100 ** 2)) / 50.0) < acceptable_error))

if __name__ == '__main__':
    unittest.main()
