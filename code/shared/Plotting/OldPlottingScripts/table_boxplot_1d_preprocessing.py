import os
import sys
code_directory =  os.path.dirname(os.path.realpath(__file__)) + '/../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)
from mongo_scripts.mongo_retrieve import mongo_query

def pull_stat(query, projection, main_stat, group_blobs_by='ex_id'):
    ''' grab those stats from the database using standard mongodb query and projection format
    and follow your dreams.
    '''
    print 'q', query
    print 'p', projection
    entries = mongo_query(query, projection)    
    labels_by_group = {}
    datasets_by_group = {}
 
    for entry in entries:
        if group_blobs_by == 'ex_id':
            ex_id = str(entry['date']) + '_' + str(entry['time_stamp'])
            group_id = ex_id
        else:
            group_id = entry[group_blobs_by]

        if group_id not in labels_by_group:
            labels_by_group[group_id] = entry   
        if group_id not in datasets_by_group: datasets_by_group[group_id] = []
        datasets_by_group[group_id].append(entry['data'][main_stat])

    # add count to the labels
    for group_id in datasets_by_group:
        count = len(datasets_by_group[group_id])
        labels_by_group[group_id].update({'count':count})
    return labels_by_group, datasets_by_group

def pull_attribute(query, projection, attribute, group_blobs_by='ex_id'):
    ''' grab those stats from the database using standard mongodb query and projection format
    and follow your dreams.
    '''
    if attribute not in projection: projection.update({attribute:1})
    print 'q', query
    print 'p', projection
    entries = mongo_query(query, projection)    
    labels_by_group = {}
    datasets_by_group = {}

    count=0
    for entry in entries:
        if group_blobs_by == 'ex_id':
            ex_id = str(entry['date']) + '_' + str(entry['time_stamp'])
            group_id = ex_id
        else:
            group_id = entry[group_blobs_by]

        if group_id not in labels_by_group:
            labels_by_group[group_id] = entry   
        if group_id not in datasets_by_group: datasets_by_group[group_id] = []
        if attribute in entry:
            datasets_by_group[group_id].append(entry[attribute])
            count+=1
    print count
    # add count to the labels
    for group_id in datasets_by_group:
        count = len(datasets_by_group[group_id])
        labels_by_group[group_id].update({'count':count})
    return labels_by_group, datasets_by_group


'''
if __name__ == '__main__':
    #query = {'data_type':'stats_size_raw','growth_medium':'solid','date':{'$gt':'20121120','$lt':'20121130'}}
    #projection =  {'age':1, 'data':1,'name':1, 'date':1}
    query = {'data_type':'stats_size_raw', 
             'strain':'N2', 
             'growth_medium':'solid',
             #'growth_medium':'liquid',
             'date':{'$gt':'20121100'},
             'age':{'$gt':'d1'}
             }
    projection = {'name':1,'date':1,'time_stamp':1, 'data':1, 'age':1,}

    entiries = mongo_query(query, projection)
    num_with_age = 0
    for ent in entiries:
        #print ent
        if 'age' in ent: num_with_age +=1 
    print 'num_with_age', num_with_age
'''
