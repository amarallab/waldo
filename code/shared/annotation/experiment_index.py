#!/usr/bin/env python

'''
Filename: experiment_index.py

Description: This class is used to get basic experimental information out of
table files and into a database friendly dictionary to use as
tags for marking experiments.

Example usages:
ei = Experiment_Attribute_Index()

ei.return_ex_ids_with_attribute(key_attribute='purpose', attribute_value='N2_aging')
ei.return_attribute_for_ex_ids(['20130423_123836', '20130410_143246'], 'pixels-per-mm')
ei.return_ex_ids_within_dates(start_date='20120300', end_date='20121000')
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import pandas as pd
import datetime
import glob

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../../')
sys.path.append(CODE_DIR)

# nonstandard imports
from settings.local import LOGISTICS

INDEX_DIR = LOGISTICS['annotation']

def Experiment_Attribute_Index2(dataset=None, index_tsv_directory=INDEX_DIR):
    ''' returns a pandas dataframe with all the annotated indicies '''
    search = os.path.join('{d}'.format(d=index_tsv_directory.rstrip('/')),'*.tsv')
    data = [pd.read_csv(f, sep='\t', index_col=0) for f in glob.iglob(search)]
    full_index = pd.concat(data)
    if dataset != None:
        full_index = full_index[full_index['dataset'] == dataset]
    return full_index

# def return_experiment_attributes(ex_id, index_tsv_directory=INDEX_DIR):
#     search = os.path.join('{d}'.format(d=index_tsv_directory.rstrip('/')), '*.tsv')
#     for ifile in glob(search):
#         df = pd.read_csv(ifile, sep='\t', index_col=0)
#     full_index = pd.concat(data)


# global default value
class Experiment_Attribute_Index(object):

    def __init__(self, index_tsv_directory=INDEX_DIR):
        """ initialize an Experiment_Attribute_Index object.

        :param index_tsv_directory: the directory that contains the index spreadsheet data in tsv files.
        """
        # make sure there is a backslash at the end of the directory name
        self.dir = '{dir}/'.format(dir=index_tsv_directory.rstrip('/'))
        #ensure_dir_exists(self.dir)
        self.files = glob.glob(self.dir + '*.tsv')
        self.attribute_index = {}
        self.ex_ids = []
        self.unflagged_ex_ids = []
        for filename in self.files:
            self.add_experiments_from_file(filename)
        if not len(self.files):
            print 'Warning: no index files found in: {dir}'.format(dir=index_tsv_directory)

    def add_experiments_from_file(self, filename):
        """ reads all ex_ids (+ metadata) from one tsv file and stores it.

        :param filename: name of tsv file being read.
        """
        with open(filename) as f:
            lines = f.readlines()
        headers = lines[0].split('\t')
        assert 'ex-id' in headers, 'ex-id not in headers for: {name}'.format(name=filename)
        assert 'vid-flags' in headers, 'vid-flags not in headers for: {name}'.format(name=filename)

        new_ex_ids, flagged_ex_ids = [], []
        for line in lines[1:]:
            cols = line.rstrip('\n').split('\t')
            ex_id = cols[headers.index('ex-id')]
            assert len(ex_id.split('_')) == 2, 'ex_id not formatted properly: {id}'.format(id=ex_id)
            assert 18 >= len(ex_id) >= 15, 'ex_id not formatted properly: {id}'.format(id=ex_id)
            self.ex_ids.append(ex_id)
            self.attribute_index[ex_id] = {}
            new_ex_ids.append(ex_id)
            for i in range(len(headers)):
                header = unicode(headers[i].strip('\n'))
                value = unicode(cols[i])
                self.attribute_index[ex_id][header] = value

            if 'vid-flags' in self.attribute_index[ex_id]:
                if len(self.attribute_index[ex_id]['vid-flags']) > 0:
                    flagged_ex_ids.append(ex_id)

        for ex_id in new_ex_ids:
            if ex_id not in flagged_ex_ids:
                self.unflagged_ex_ids.append(ex_id)

    def return_attributes_for_ex_id(self, ex_id):
        """ return a dict of attributes for an ex_id.

        :param ex_id: ex
        :return: a dict of attributes
        """
        return self.attribute_index.get(ex_id, None)

    def return_ex_ids_within_dates(self, start_date, end_date):
        """
        :param start_date: a string of 8 digits. four for year, two for month, two for day.  ex.'20120914'
        :param end_date: a string of 8 digits, like start_date
        :return: a list of ex_ids that have times falling in between start date and end date.
        """
        assert type(start_date) == type(end_date) == str, 'start and end dates must be strings'
        assert len(start_date) == len(end_date) == 8, 'start and end dates must be 8 char long'
        return [ex_id for ex_id in self.unflagged_ex_ids
                if (int(start_date) <= int(ex_id.split('_')[0]) <= int(end_date))]


    def return_ex_ids_with_attribute(self, key_attribute, attribute_value):
        """ returns a list of ex_ids that have a key_attribute which is equal to an
        attribute value.

        :param key_attribute: string with the name of the desired attribute 
        :param attribute_value: string with the value of the desired attribute
        """
        return [ex_id for ex_id in self.attribute_index
                if self.attribute_index[ex_id].get(key_attribute, None) == attribute_value]


    def return_attribute_for_ex_ids(self, ex_ids, attribute):
        """ returns a list of attributes the same length and order as the input list of ex_ids

        :param ex_ids: a list of ex_ids for which you want some attribute
        :param attribute: a string denoting which attribute you would like for each of the ex_ids
        """
        return [self.attribute_index[ex_id].get(attribute, None) for ex_id in ex_ids]

def list_ex_ids_with_raw_data(inventory_directory):
    ''' make list of ex_ids present in the data directory on the cluster.

    :param inventory_directory: directory to search for ex_id data.
    '''
    search_path = inventory_directory + '*'
    ex_ids = []

    for entry in glob.glob(search_path):
        dirname = entry.split('/')[-1]
        if os.path.isdir(entry) and len(dirname) == 15:
            ex_ids.append(dirname)

    if len(ex_ids) < 5:
        print 'Warning: not many ex_id directories found'
        print 'search path for raw data is ({sp})'.format(sp=search_path)
        print '{N} ex_ids present'.format(N=len(ex_ids))
    return ex_ids

def ex_id_to_datetime(ex_id):
    ''' converts an experiment id to a datetime object '''
    parts = ex_id.split('_')
    if len(parts) != 2:
        print 'Error: something is off with this ex_id', ex_id
        return None
    ymd, hms = parts
    year, month, day = map(int, [ymd[:4], ymd[4:6], ymd[6:]])
    h, m, s = map(int, [hms[:2], hms[2:-2], hms[-2:]])
    return datetime.datetime(year, month, day, h, m, s)

def organize_plate_metadata(ex_id):
    ei = Experiment_Attribute_Index()

    m = ei.return_attributes_for_ex_id(ex_id)
    label = m.get('label', 'label')
    sub_label = m.get('sublabel', 'set')
    sub_label = '{l}-{sl}'.format(l=label, sl=sub_label)
    pID = m.get('plate-id', 'set B')
    day = m.get('age', 'A0')

    recording_time = ex_id
    plating_time = m.get('l1-arrest', None)
    #print 'plated at:', plating_time
    #print 'recorded at:', recording_time
    hours = 0
    if recording_time and plating_time:
        t0 = ex_id_to_datetime(plating_time)
        t1 = ex_id_to_datetime(recording_time)
        hours = (t1 - t0).total_seconds()/3600.
    #age = '{et} - {pt}'.format(et=recording_time, pt=plating_time)
    #for i in m:
    #    print i
    return hours, label, sub_label, pID, day




if __name__ == '__main__':
    ei = Experiment_Attribute_Index2()
    # examples of possible usages
    print len(ei.index), 'total'
    #print len(ei.unflagged_ex_ids), 'unflagged'
    '''
    print ei.return_ex_ids_with_attribute(key_attribute='purpose', attribute_value='N2_aging')
    print ei.return_attribute_for_ex_ids(['20130423_123836', '20130410_143246', '20130413_150111', '20130325_152726'], 'pixels-per-mm')
    print ei.return_ex_ids_within_dates(start_date='20120300', end_date='20121000')
    '''