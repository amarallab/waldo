#!/usr/bin/env python

'''
Filename: google_spreadsheet_interface.py
Description: interface between google docs and my worm spreadsheets.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

from itertools import izip
import json
import gdata.spreadsheet.service
import gdata.service
import gdata.spreadsheet

try:
    from xml.etree import ElementTree
except ImportError:
    from elementtree import ElementTree


def truncate(content, length=15, suffix='...'):
    if len(content) <= length:
        return content
    else:
        return content[:length] + suffix

class Spreadsheet_Interface:

    def __init__(self, email, password, row_id='ex-id'):
        self.gd_client = gdata.spreadsheet.service.SpreadsheetsService()
        self.gd_client.email = email
        self.gd_client.password = password
        self.gd_client.source = 'Spreadsheets GData Sample'
        self.gd_client.ProgrammaticLogin()
        self.row_id = row_id
        
    def index_spread_sheets(self):
        feed = self.gd_client.GetSpreadsheetsFeed()
        sheet_to_id = {}
        for item in feed.entry:
            title = item.title.text.split('/')[-1]
            spreadsheet_id = item.id.text.split('/')[-1]
            #print title, spreadsheet_id
            sheet_to_id[title] = spreadsheet_id
        #self.sheet_to_id = sheet_to_id
        worksheet_ids_by_sheet_name = {}

        for sheet_name, sheet_id in sheet_to_id.iteritems():
            worksheet_ids_by_sheet_name[sheet_name] = {}
            feed = self.gd_client.GetWorksheetsFeed(sheet_id)
            for i in feed.entry:
                title = i.title.text.split('/')[-1]
                worksheet_id = i.id.text.split('/')[-1]
                #print title, worksheet_id
                worksheet_ids_by_sheet_name[sheet_name][title] = worksheet_id
        #self.worksheet_ids = worksheet_ids_by_sheet_name
        return sheet_to_id, worksheet_ids_by_sheet_name

    def _StringToDictionary(self, row_data):
        row_dic = {}
        for param in row_data.split(','):
            temp = param.split(':')
            if len(temp) >= 2:
                row_dic[temp[0].strip()] = temp[1].strip()
            else:
                print 'warning', temp, param
        return row_dic

    def names_to_keys(self, spreadsheet, worksheet):
        key, wksht_id = None, None
        for item in self.gd_client.GetSpreadsheetsFeed().entry:
            if spreadsheet == item.title.text.split('/')[-1]:
                key = item.id.text.split('/')[-1]
                feed = self.gd_client.GetWorksheetsFeed(key=key)
                for i in feed.entry:
                    if worksheet == i.title.text.split('/')[-1]:
                        wksht_id = i.id.text.split('/')[-1]
        if not key: print 'Warning!:', spreadsheet, 'not found'
        if not wksht_id: print 'Warning!:', worksheet, 'not found'
            
        return key, wksht_id

    def upload_sheet(self, headers, rows, spreadsheet, worksheet):
        """
        :param headers: list of column headers that will appear in the first row.
        :param rows:
        :param spreadsheet: name of the google-spreadsheet (str)
        :param worksheet: name of worksheet inside google-spreadsheet (str)
        """
        # make sure that row_id is the first item in the headers list
        # this will make later identification possible
        row_id = self.row_id
        if str(row_id) != str(headers[0]):
            headers = [row_id] + headers

        # find address of sheet + worksheet
        sheet_to_id, worksheet_ids = self.index_spread_sheets()
        if worksheet in worksheet_ids[spreadsheet]:
            worksheet_entry = self.gd_client.GetWorksheetsFeed(key=sheet_to_id[spreadsheet],
                                                               wksht_id=worksheet_ids[spreadsheet][worksheet])
            # TODO: if deleting everything, google does not allow deletion of last worksheet
            # make this work around that
            self.gd_client.DeleteWorksheet(worksheet_entry=worksheet_entry)

        self.gd_client.AddWorksheet(title=worksheet, key=sheet_to_id[spreadsheet],
                                    row_count=len(rows), col_count=len(headers))

        
        key, wksht_id = self.names_to_keys(spreadsheet=spreadsheet, worksheet=worksheet)
        # write header row on the worksheet
        for i, col_name in enumerate(headers, start=1):
            entry = self.gd_client.UpdateCell(row=1, col=i, inputValue=str(col_name), key=key, wksht_id=wksht_id)            
            assert isinstance(entry, gdata.spreadsheet.SpreadsheetsCell)
        
        for row in rows:
            #for i in row:
            #    assert i in headers, '{col} is not in headers\n{h}'.format(col=i, h=headers)
            #    assert isinstance(row[i], str), '{col} is not in string format\n{h}'.format(col=i, h=headers)
            self.gd_client.InsertRow(row_data=row, key=key, wksht_id=wksht_id)
            

    def download_sheet(self, spreadsheet, worksheet):
        """
        For a google-spreadsheet returns a tuple containing (1) the list of headers, (2) the list of row IDs (3) a
        list of dictionaries containing the contents of each row by header. empty boxes are not part of

        :param spreadsheet: name of the google-spreadsheet (str)
        :param worksheet: name of worksheet inside google-spreadsheet (str)
        :return: tuple of headers (list of strings), row names (list of strings), and rows (list of dicts)
        """
        key, wksht_id = self.names_to_keys(spreadsheet=spreadsheet, worksheet=worksheet)

        # get headers, use cell feed. cant figure it out with list_feed
        feed = self.gd_client.GetCellsFeed(key=key, wksht_id=wksht_id)
        col_header = []
        for entry in feed.entry:
            if entry.cell.row == '1':
                col_header.append((entry.cell.col, entry.cell.text))
        if len(col_header) == 0:
            print spreadsheet, worksheet, 'did not have headers'
            return [], [], []
        cols, headers = zip(*col_header)
        # use list feed to get the rest.
        feed = self.gd_client.GetListFeed(key=key, wksht_id=wksht_id)
        rows = []
        row_names = []
        for i, entry in enumerate(feed.entry):
            row_names.append(entry.title.text)
            rows.append(self._StringToDictionary(entry.content.text))
        
        return headers, row_names, rows

    def download_all_worksheets(self, sheet_name, write_tsvs=True, save_dir=''):
        """
        Returns a dictionary containing the contents of all the worksheets within a particular google-spreadsheet.
        Has an option to write all worksheets to json files.

        :param sheet_name: name of the google-docs sheet to download (string)
        :param write_tsvs: toggle that will write a json for each worksheet (bool)
        :param save_dir: if write_tsvs, this string specifies the directory jsons should be written to.
        :return:
        """
        sheet_to_id, worksheet_ids = self.index_spread_sheets()
        assert sheet_name in sheet_to_id

        worksheets = worksheet_ids[sheet_name]
        sheet_dict = {}
        for i, worksheet in enumerate(worksheets):
            headers, rn, r = self.download_sheet(sheet_name, worksheet)

            this_sheet = {}
            for row_id, attributes in izip(rn, r):
                this_sheet[row_id] = attributes
                this_sheet[row_id][self.row_id] = row_id
            sheet_dict[worksheet] = this_sheet

            if write_tsvs:
                #print 'writing {worksheet} file to {dir}'.format(worksheet=worksheet, dir=save_dir)
                save_name = '{dir}/{name}.tsv'.format(dir=save_dir.rstrip('/'),
                                                      name=worksheet)
                with open(save_name, 'w') as f:
                    f.write('\t'.join(headers) + '\n')
                    for row_id in sorted(this_sheet):
                        f.write('\t'.join([this_sheet[row_id].get(h, '')
                                           for h in headers]) + '\n')
            #for sheet_name, contents in sheet_dict.iteritems():
            #print 'writing {name}'.format(name=save_name)
            #json.dump(contents, open(save_name, 'w'), indent=4, sort_keys=True)
        return sheet_dict

    def pull_scaling_factors(self, sheet_name):
        print 'pulling scaling_factors'
        scaling_factors = {}
        sheet_to_id, worksheet_ids = self.index_spread_sheets()
        cameras = worksheet_ids[sheet_name]
        for camera in cameras:        
            scaling_factors[camera] = []
            headers, row_names, rows = self.download_sheet(spreadsheet='scaling-factors', worksheet=camera)
            #print headers, row_names, rows
            for row in rows:
                array_row = []
                for h in headers:
                    try:
                        array_row.append(row[h])
                    except:
                        print 'warning', h, 'not in',  row
                scaling_factors[camera].append(array_row)
        #print scaling_factors[camera]
        return scaling_factors
