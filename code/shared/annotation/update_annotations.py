#!/usr/bin/env python

'''
Filename: update_annotations.py

Description: This script updates experiment annotation by (1) downloading info from google-spreadsheets to the local
computer and (2) uploading basic information from undocumented experiments to the google-spreadsheet.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import glob
import os
import sys
import bisect

# path definitinos
CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(SHARED_DIR)
sys.path.append(CODE_DIR)

# nonstandard imports
from settings.local import SPREADSHEET, LOGISTICS
from wio.file_manager import ensure_dir_exists
from wio.google_spreadsheet_interface import Spreadsheet_Interface

# Globals
USER = SPREADSHEET['user']
PASSWORD = SPREADSHEET['password']
ANNOTATION_SHEET = SPREADSHEET['spreadsheet']
SCALEING_SHEET = SPREADSHEET['scaling-factors']
ROW_ID = SPREADSHEET['row-id']
HEADERS = SPREADSHEET['columns']

DEFAULT_DATA_DIR = LOGISTICS['filesystem_data']
DEFAULT_LOGISTICS_DIR = LOGISTICS['inventory']
DEFAULT_SAVE_DIR = LOGISTICS['annotation']

def inventory_data_directories(base_data_dir=DEFAULT_DATA_DIR):
    """
    returns a dict with all year-months (ex. '2013-05') in which data was collected.
    each year-month is a key
    for another dictionary of ex_ids and their corresponding data directory.

    :param base_data_dir: the directory that should be inventoried
    :return: dictionary of year-months containing dictionaries of ex_ids
    """
    files_and_dirs = glob.glob(base_data_dir+ '*')
    dirs_by_yearmonth = {}
    for entry in files_and_dirs:
        ex_id = entry.split('/')[-1]

        if os.path.isdir(entry) and len(ex_id) == 15:
            date, time_stamp = ex_id.split('_')
            yearmonth = '{year}-{month}'.format(year=date[:4], month=date[4:6])
            if yearmonth not in dirs_by_yearmonth:
                dirs_by_yearmonth[yearmonth] = {}
            if ex_id not in dirs_by_yearmonth[yearmonth]:
                dirs_by_yearmonth[yearmonth][ex_id] = entry
    return dirs_by_yearmonth

def update_annotation_worksheet(data_ex_ids, annotated_ex_ids, ex_ids_to_add, ex_ids_to_remove, source_computers, scaling_factors):
    """
    This returns spreadsheet data in the appropriate form for uploading using a Spreadsheet_Interface from
    google_spreadsheet_interface.py. This format is a list of dictionaries with headers corresponding to column names.

    :param data_ex_ids: a dictionary of ex_ids and the path to their directory.
    :param annotated_ex_ids: a dictionary of ex_ids and the dictionary of values from the google-spreadsheet.
    :param ex_ids_to_add: a list of ex_ids that should be added/updated
    :param ex_ids_to_remove: a list of ex_ids that should not be included
    :param source_computers: a dict linking ex_ids to their source computers
    :param scaling_factors: a nested dict needed to link ex_ids to their appropriate scaling-factors
    :return:
    """
    updated_sheet = []
    for ex_id in ex_ids_to_remove:
        del annotated_ex_ids[ex_id]

    for ex_id in ex_ids_to_add:
        ex_dir = data_ex_ids[ex_id]
        if ex_id not in annotated_ex_ids:
            annotated_ex_ids[ex_id] = {}
        annotated_ex_ids[ex_id].update(get_attributes_for_dir(ex_dir=ex_dir, sources=source_computers,
                                                              scaling_factors=scaling_factors))
        #print annotated_ex_ids[ex_id]
    for ex_id in sorted(annotated_ex_ids.keys()):
        updated_sheet.append(annotated_ex_ids[ex_id])
    return updated_sheet

def get_source_computers(inventory_path=DEFAULT_LOGISTICS_DIR):
    """
    Reads inventory files and returns a dictionary of ex_ids as keys and the source as values.

    :param inventory_path: path to the inventory files used to keep track of source computers. This is usually stored
     in settings.py.
    """
    source_computers = {}
    for ci in glob.glob(inventory_path + 'camera*'):
        camera = ci.split('-')[-1].rstrip('.txt')
        with open(ci, 'r') as f:
            ex_ids = [line.strip() for line in f.readlines()]
        for ex_id in ex_ids:
            source_computers[ex_id] = camera
    return source_computers

def update_main(update_list=[], update_all=False, overwrite=False,
                remove_missing=True, save_dir=DEFAULT_SAVE_DIR):
    """
    A two step process that (1) writes all currently annotated data (in google-docs) to json files and (2) Updates
    google-docs to contain entries for all raw data files.

    :param update_list: a list of dates that should be updated. if specified only dates in list will be examined.
    :param update_all: update spreadsheet even if no recordings appear to be missing
    :param overwrite: if any miss-match occurs between annotation and raw data, start spreadsheet over from scratch.
    :param remove_missing: remove rows from the google-docs if they are not present in local raw data
    """
    # step1: initiate connection to google-docs and download/write all annotations
    si = Spreadsheet_Interface(email=USER, password=PASSWORD, row_id=ROW_ID)
    ensure_dir_exists(save_dir)
    all_annotated_sheets = si.download_all_worksheets(sheet_name=ANNOTATION_SHEET,
                                                      write_tsvs=True, save_dir=save_dir)

    # Only initialize these if updates are required.
    source_computers, scaling_factors = None, None

    # step2: go through all
    dirs_by_yearmonth = inventory_data_directories()
    print 'writing annotation files to: {p}'.format(p=os.path.abspath(save_dir))
    for yearmonth in sorted(dirs_by_yearmonth):
        data_ex_ids = dirs_by_yearmonth[yearmonth]
        annotated_ex_ids = all_annotated_sheets.get(yearmonth, {})

        # only update if there is a missmatch between ex_ids with data and ex_ids that have been annotated
        if update_all or (yearmonth in update_list) or (set(annotated_ex_ids.keys()) != set(data_ex_ids.keys())):

            if not source_computers:
                source_computers = get_source_computers()
            if not scaling_factors:
                scaling_factors = si.pull_scaling_factors(sheet_name=SCALEING_SHEET)

            # determine how many ex_ids to add and how many to remove.
            ex_ids_to_add, ex_ids_to_remove = [], []
            if overwrite:
                ex_ids_to_add = data_ex_ids.keys()
            else:
                ex_ids_to_add = [ex_id for ex_id in data_ex_ids if ex_id not in annotated_ex_ids]
            if remove_missing:
                ex_ids_to_remove = [ex_id for ex_id in annotated_ex_ids if ex_id not in data_ex_ids]

            # perform the update
            updated_rows = update_annotation_worksheet(data_ex_ids, annotated_ex_ids, ex_ids_to_add, ex_ids_to_remove,
                                                       source_computers, scaling_factors)
            si.upload_sheet(headers=HEADERS, rows=updated_rows, spreadsheet=ANNOTATION_SHEET, worksheet=yearmonth)

            # inform the user of the update.
            msg = ''
            if ex_ids_to_add:
                msg = '{msg} added {N},'.format(msg=msg, N=len(ex_ids_to_add))
            if ex_ids_to_remove:
                msg = '{msg} removed {N} '.format(msg=msg, N=len(ex_ids_to_remove))
            print 'updated {yearmonth}: {msg}'.format(yearmonth=yearmonth, msg=msg)
        else:
            print yearmonth, len(data_ex_ids), 'current'


def get_attributes_for_dir(ex_dir, sources, scaling_factors):
    """
    creates a new index dictionary with several automatically generated attributes for data in a given directory.

    :param ex_dir: the directory containing data for a single recording.
    :return: a dictionary with several attributes of that experiment.
    """

    ex_id = ex_dir.split('/')[-1]
    #print 'anotating', ex_id, '\n', ex_dir
    name, vid_duration = parse_summary_file(ex_dir)
    source = sources.get(ex_id, '?')
    scaling_factor = str(find_scaling_factor(ex_id, source, scaling_factors))
    return {'ex-id': ex_id,
            'vid-flags': '',
            'name': name,
            'vid-duration': str(vid_duration),
            'num-blobs-files': str(len(glob.glob(ex_dir + '/*.blobs'))),
            'num-images': str(len(glob.glob(ex_dir + '/*.png'))),
            'source-camera': source,
            'pixels-per-mm': scaling_factor}


def find_scaling_factor(ex_id, source, scaling_factors, verbose=False):
    """
    this returns the appropriate scaling factor for an ex_id, given a dictionary of source computers,
    and a nested dictionary of scaling factors.

    :param ex_id: experiment id (str)
    :param source: dictionary of ex_ids and their source computer
    :param scaling_factors: dictionary of scaling factors by source computer and then ex_id.
    :return: a numerical scaling factor denoting the number of pixels per mm
    """

    scaling_factor = 1
    if source in scaling_factors:
        start_dates, factors = zip(*scaling_factors[source])

        if ex_id in start_dates:
            i = start_dates.index(ex_id)

        else:
            i = bisect.bisect_left(start_dates, ex_id) - 1

        if i >= len(factors):
            scaling_factor = factors[-1]
        else:
            scaling_factor = factors[i]

        if verbose:
            for j, s in enumerate(scaling_factors[source]):
                if j == i:
                    print s[0], s[1], '<--', ex_id
                else:
                    print s[0], s[1]

    return scaling_factor

def test_scaling_factors(ex_id='20130718_154612'):
    """
    This finds and displays the source_computer and scaling_factor data for one ex_id.

    :param ex_id: experiment id string
    """
    #ex_id = '20130702_150904'
    #ex_id = '20120331_165736'
    #ex_id = '20130826_152550'
    #ex_id = '20120503_175416'
    #ex_id = '20130904_131903'

    si = Spreadsheet_Interface(email=USER, password=PASSWORD)
    scaling_factors = si.pull_scaling_factors(sheet_name=SCALEING_SHEET)
    source_computers = get_source_computers()
    source = source_computers.get(ex_id, '?')

    scaling_factor = find_scaling_factor(ex_id=ex_id, source=source, scaling_factors=scaling_factors, verbose=True)
    
    print 
    print '{ex_id} {source} {sf}'.format(ex_id=ex_id, source=source, sf=scaling_factor)

def parse_summary_file(ex_dir):
    """
    Returns a file_name and the length of

    :param ex_dir: directory containing data for a particular experiment.
    """
    summary_files = glob.glob(ex_dir + '/*.summary')
    file_name = ''
    record_length = 0
    if len(summary_files) >= 1:
        summary_file = summary_files[0]
        assert os.path.isfile(summary_file)
        # parse the summary file name
        file_name = summary_file.split('/')[-1].split('.summary')[0]

        # if the summary file contains info, read length and median blobs
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        if len(lines) >= 1:
            record_length = int(float(lines[-1].split()[1]))
    return file_name, record_length

if __name__ == '__main__':    
    #update_main(update_all=False)
    #find_ex_ids_to_update(update_list=['2012-10', '2012-11', '2012-12'])
    #find_ex_ids_to_update(update_list=['2013-01', '2013-02', '2013-03', '2013-04', '2013-05', '2013-06', '2013-07', '2013-08'])

    #update_main(update_all=True)
    #update_main(update_list=['2013-09']) # thermotolerance
    update_main(update_list=['2013-03', '2013-04']) # N2_aging
    #update_main(update_list=['2013-06', '2013-07']) # disease models
    #update_main(update_list=['2013-12']) # copas lifespan
    #test_scaling_factors()
