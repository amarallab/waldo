#!/usr/bin/env python

'''
Filename: import_test_worms.py
Discription: This script contains all necessary functions to import jsons containing test data into the basic
input types used for all data processing code.

The jsons are not generated using this code, but they contain dictionaries using times as keys
(bson friendly formatting) and lists of x,y tuples that specify points along a worm's outline as values)
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import json
import os
import sys
from glob import glob
import numpy as np

# path specifications
test_directory = os.path.dirname(os.path.realpath(__file__)) 
project_directory = test_directory + '/../../'

test_data_dir = test_directory + '/data/'
print test_data_dir
assert os.path.exists(test_data_dir)
assert os.path.exists(project_directory), 'project directory not found'
sys.path.append(project_directory)

# nonstandard imports
from database.mongo_insert import insert_blob_entries
from Encoding.decode_outline import encode_outline
from GeometricCalculations.matrix_and_point_operations import close_outline_border
from scipy.ndimage.morphology import binary_fill_holes

# globals
test_types = {
'SquareTests': ('00000000_000001', 'moves in partial square. worms oriented at different angles to test measures.'),
'BreakTests': ('00000000_000003', 'circular moving segment. different bad segments inserted to test breaks.'),
}

assert os.path.exists(test_directory + '/settings/fake_dict.json')
testworm_attribute_dict = json.load(open(test_directory + '/settings/fake_dict.json', 'r'))


def import_tests_of_type(test_type, import_data=True, verbose=False, **kwargs):
    """
    This function imports all the jsons containing test data that are in the directory specified by the
    'test_types' variables. Similar data files have been grouped together. This is reflected by putting
    all the worms from a similar type in the database with the same experiment id.

    :param test_type: this is a list of
    :param import_data: a boolean value that specifies if the jsons should be imported or not.
    :return: a list of blob_ids of the specified test_type
    """
    assert test_type in test_types

    # make sure that test jsons exist for this type
    test_data_dir = test_directory + 'Data/' + test_type
    test_jsons = glob(test_data_dir + '/*.json')
    if len(test_jsons) == 0:
        print 'no test json files found in', test_data_dir

    # 
    ex_id, description = test_types[test_type]
    testworm_attribute_dict['ex_id'] = ex_id

    if verbose:
        print 'inserting test worms'
        print ex_id, description
    blob_ids = []
    for i, tjson in enumerate(sorted(test_jsons), start=1):
        blob_id = ('00000%i' % i)[-5:]
        blob_id = ex_id + '_' + blob_id
        blob_ids.append(blob_id)
        outline_timeseries = json.load(open(tjson, 'r'))

        #timeseries in json was all floats. convert to int to stay consistency
        for o in outline_timeseries.keys():
            outline_timeseries[o] = [(int(x), int(y)) for (x, y) in outline_timeseries[o]]
        if import_data:
            if verbose:
                print 'inserting', blob_id
            push_test_worms_into_db(blob_id, outline_timeseries, description, testworm_attribute_dict, **kwargs)
    return blob_ids

def outline_points_to_encoded_outline(outline_timeseries):
    """
    returns an encoded version of the outline timedict to save space and match the format we are using to aquire raw
    outline data.

    :param outline_timeseries: a timedict that has lists of points (x,y tuples)
    :return: a timedict that hases the list of points encoded as a string.
    """
    encoded_outline_timedict = {}
    for time_key in outline_timeseries:
        points = outline_timeseries[time_key]
        encoded_outline_timedict[time_key] = encode_outline(points)
    return encoded_outline_timedict

def outline_points_to_size(outline_timeseries):
    """
    takes a time dictionary of outline points and returns a timedict with the same keys as input,
    but each value is not an int specifying how many pixels are contained within that filled in outline.

    :param outline_timeseries: a timedict that has lists of points (x,y tuples)
    :return: timedict with the same keys as input, but containing ints specifying size of shape.
    """
    size_timedict = {}
    for time_key in outline_timeseries:
        outline = outline_timeseries[time_key]
        outline = close_outline_border(outline)
        xs, ys = zip(*outline)
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        boxed_xs = [(x - min(xs) + 1) for x in xs]
        boxed_ys = [(y - min(ys) + 1) for y in ys]
        outline_matrix = np.zeros([x_range + 2, y_range + 2], dtype=int)
        for x, y in zip(boxed_xs, boxed_ys):
            outline_matrix[x][y] = 1
        filled_matrix = binary_fill_holes(outline_matrix)
        size = int(sum(sum(filled_matrix)))
        size_timedict[time_key] = size
    return size_timedict

def outline_points_to_centroid_xy(outline_timeseries):
    """
    takes a time dictionary of outline points and returns a timedict with the same keys as input,
    but each value is now one x,y tuple specifying the coordinates of the shapes centroid.

    :param outline_timeseries: a timedict that has lists of points (x,y tuples)
    :return: timedict with the same keys as input, but containing one x,y tuple of the centroid position
    """
    xy_timedict = {}
    for time_key in outline_timeseries:
        x, y = zip(*outline_timeseries[time_key])
        xy_timedict[time_key] = (np.mean(x), np.mean(y))
    return xy_timedict

def push_test_worms_into_db(blob_id, outline_timeseries, description, test_attributes, **kwargs):
    """
    This function pushes the data for one test blob_id into the database in all the standard formats.

    :param blob_id: identification string used to keep track of this data
    :param outline_timeseries: the data stored in the json file.
    :param description: a short string explaining what
    :param test_attributes: metadata for database queries that mimics the format found in real data.
    """
    midline_estimates = [(len(outline) / 2.2) for outline in outline_timeseries.values()]
    test_attributes.update({'midline_median': np.median(midline_estimates)})
    # 'size_median' -- we have real size data, so we actually calculate this
    size_dict = outline_points_to_size(outline_timeseries)
    sizes = [size for size in size_dict.values()]
    #for (i, size) in size_dict: print i, size
    test_attributes.update({'size_median': np.median(sizes)})
    # 'local_blob_id' -- last set of numbers from blob_id 
    test_attributes.update({'local_blob_id': blob_id.split('_')[-1]})
    # 'start_time' and 'stop_time' -- get from outline_timeseries start and stop times
    times = sorted([float(t.replace('?', '.')) for t in outline_timeseries])
    test_attributes.update({'start_time': times[0], 'stop_time': times[-1], 'duration': times[-1] - times[0]})

    # 'bl_dist'-- this is a pain to calculate right here so my test data will just say 5
    metadata_entry = {'data': None,
                      'data_type': 'metadata',
                      'description': 'test worm. ' + description,
                      'blob_id': blob_id, }

    outline_entry = {'data': outline_points_to_encoded_outline(outline_timeseries=outline_timeseries),
                     'data_type': 'encoded_outline',
                     'description': 'test worm. ' + description,
                     'blob_id': blob_id, }

    size_entry = {'data': size_dict,
                  'data_type': 'size_raw',
                  'description': 'test worm. ' + description,
                  'blob_id': blob_id, }

    xy_entry = {'data': outline_points_to_centroid_xy(outline_timeseries),
                'data_type': 'xy_raw',
                'description': 'test worm. ' + description,
                'blob_id': blob_id, }

    metadata_entry.update(test_attributes)
    outline_entry.update(test_attributes)
    size_entry.update(test_attributes)
    xy_entry.update(test_attributes)

    insert_blob_entries([metadata_entry, outline_entry, size_entry, xy_entry], **kwargs)


if __name__ == '__main__':
    import_tests_of_type('SquareTests', import_data=True)
    #import_tests_of_type('BreakTests')
    
