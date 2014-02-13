#!/usr/bin/env python

# WARNING DEPRECIATED
'''
Filename: pull_db_results.py
Description:
'''

import os
import sys

# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(SHARED_DIR)
sys.path.append(CODE_DIR)

# nonstandard imports
from database.mongo_retrieve import mongo_query
from PrincipalComponents.compute_pca import write_component_report

id_cols = ['blob_id', 'strain', 'age', 'food', 'growth_medium']

def pull_results(file_id, outfolder, query={}, **kwargs):
    blob_entries = mongo_query(query, col='result_collection', **kwargs)
    ids=[]
    id_rows = []
    data_rows=[]
    data_categories=[]
    num_stats = 345

    for e in blob_entries[:]: 
        itemcheck = True
        for i in id_cols: 
            if i not in e: itemcheck = False
        if (len(e['data']) == num_stats) and itemcheck:
            ids.append(e['blob_id'])
            
            #id_rows.append([('%s:?:%s'%(str(i), str(e[i]))) for i in e if (i != 'data') ])
            id_rows.append([e[i] for i in id_cols])

            row = []
            if len(data_categories) == 0:
                data_categories = sorted(e['data'])
            else:
                for dc,ec in zip(data_categories, sorted(e['data'])):
                    assert dc == ec
            for stat_key in data_categories:
                row.append(e['data'][stat_key])
            data_rows.append(row)

    write_data_matrix(data_rows, file_id, outfolder)
    write_row_id_file(id_rows, file_id, outfolder)
    write_column_id_file(data_categories, file_id, outfolder)
    print len(data_rows), 'data rows written'

def write_row_id_file(id_rows, file_id, outfolder):
    savename = outfolder + '/' + file_id + '_blob_traits.txt'
    f = open(savename, 'w')
    headers = str(id_cols[0])
    #for i in id_cols[1:]: headers += '\t' + i        
    f.write(headers + '\n')
    #print headers
    for id_row in id_rows:
        line = id_row[0]
        for j in id_row[1:]: 
            line += '\t' + j
        line += '\n'
        f.write(line)
    f.close()

def write_column_id_file(col_ids, file_id, outfolder):
    savename = outfolder + '/' + file_id + '_col_stats.txt'
    #print savename
    f = open(savename, 'w')
    for c in col_ids: 
        #print c
        f.write(c + '\n')
    f.close()

def write_data_matrix(data_rows, file_id, outfolder):
    savename = outfolder + '/' + file_id + '_data_rows.txt'
    f = open(savename, 'w')
    for data_row in data_rows: 
        line = str(data_row[0])
        for datum in data_row:
            line += '\t' + str(datum)
        f.write(line + '\n')
    f.close()

def read_data_matrix(filename):
    data_rows = []

    for row in open(filename, 'r'):
        #print 'a', row
        #print
        #for i in row.rstrip('\n').split('\t'):
        #    print len(i)
        #print 'c', [float(i) for i in row.rstrip('\n').split('\t') if len(i) >0]
        data_rows.append(map(float, row.split('\t')) )
    return data_rows
    
def compute_pca(data_row_file_list, outfolder):
    full_data_rows = []
    len_drows = []
    for data_row_file in data_row_file_list:
        data_rows = read_data_matrix(data_row_file)
        print data_row_file, len(data_rows)
        len_drows.append(len(data_rows))
        for dr in data_rows: full_data_rows.append(dr)
    ids = [i for i in xrange(len(full_data_rows))]
    write_component_report(ids, full_data_rows, outfolder, overwrite_folder=True)
    return len_drows

if __name__ == '__main__':
    # part1, run a bunch of times
    outfolder = 'Mar4-Set'
    #file_id = 'W4_d4'
    #query = {'strain':'W4', 'age':'d4', 'duration':{'$gt':300}, 'size_median':{'$gt':200}}
    file_id = 'unc-58_d4'
    query = {'strain':'unc-58', 'age':'d4', 'duration':{'$gt':60}} #, 'size_median':{'$gt':60}}
    query = {'strain':'unc-58', 'age':'d4'} #, 'size_median':{'$gt':60}}
    #pull_results(file_id, outfolder, query)

    data_row_file_list = [outfolder + '/N2_d4_data_rows.txt',
                          outfolder + '/W2_d4_data_rows.txt',
                          outfolder + '/W4_d4_data_rows.txt',
                          #outfolder + '/unc-58_d4_data_rows.txt',
                          #outfolder + '/unc-58_d3_data_rows.txt',
                          ]
    pca_dir = outfolder+'/Combined-PCA'
    len_drows = compute_pca(data_row_file_list, pca_dir)

