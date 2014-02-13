'''
Author: Peter Winter
Date: 
Description:
a plot that packs a whole bunch of boxplots for different conditions right next to one another.
shows the spread of one variable for many conditions

'''
from pylab import *
import random

from Shared.Code.Plotting.OldPlottingScripts.table_boxplot_1d_preprocessing import pull_stat


def sort_ex_ids_by_label(labels_by_ex_id, sort_key):
    key_id_pairs = []
    for ex_id in labels_by_ex_id:
        label = labels_by_ex_id[ex_id]
        if sort_key in label:
            key = label[sort_key]
            key_id_pairs.append((key, ex_id))
        else: print label
    sorted_keys, sorted_ex_ids = zip(*sorted(key_id_pairs, reverse=True))
    return sorted_ex_ids

def table_boxplot(labels_by_ex_id, datasets_by_ex_id, make_boxplot=True, x_label='value', scatter=0.2):
    ''' make the plot.
    '''

    figure()
    labels = []
    pos = []
    max_shown = len(datasets_by_ex_id)
    #max_shown = 20
    boxdata = []
    #sorted_ex_ids_by_age = sort_ex_ids_by_label(labels_by_ex_id, 'age')

    #for i, ex_id in enumerate(sorted_ex_ids_by_age[:max_shown]):
    for i, ex_id in enumerate(sorted(datasets_by_ex_id)[:max_shown]):
        name = str(labels_by_ex_id[ex_id]['name'])
        count = str(labels_by_ex_id[ex_id]['count'])
        age = str(labels_by_ex_id[ex_id]['age'])
        tag = '%s    count:%s' %(age, count)
        
        x = datasets_by_ex_id[ex_id]
        ypos = i+1
        y = [ypos+random.random()*scatter-0.5*scatter for j in range(len(x))]
        text(520,ypos-0.2, tag)
        plot(x,y, marker='.', color='k', ls='0', alpha=0.1)#, label=str(leg))#, legend=legends[i])
        #plot(x,y, marker='.', color='b', ls='0')#, label=str(leg))#, legend=legends[i])
        boxdata.append(x)
    #if len(boxdata) >= 2: boxplot(boxdata,0,'rs',0)    
        
    ylim([0,max_shown+1])
    #xlim([0,680])
    xlabel(x_label)
    yticks([])
    show()

if __name__ == '__main__':
    

    input_data_type = 'mwt_king_raw'
    main_stat = 'median'
    #attribute = 'midline_median'

    #all_ex_ids = return_all_ex_ids()
    query = {
             'data_type':input_data_type, 
             #'data_type':'metadata', 
             'strain':'N2', 
             #'growth_medium':'solid',
             #'midline_median':{'$gt':0.1},
             'growth_medium':'liquid',
             #'date':{'$gt':'20121100'},
             #'age':{'$gt':'d1'}
             }
    projection = {'name':1,'date':1,'time_stamp':1, 'data':1, 'age':1,}
    group_by = 'ex_id'
    group_by = 'age'

    labels_by_ex_id, datasets_by_ex_id = pull_stat(query, projection, main_stat, group_by)
    #labels_by_ex_id, datasets_by_ex_id = pull_attribute(query, projection, attribute, group_by)
    age_count = 0
    for i in labels_by_ex_id:
        if 'age' in labels_by_ex_id[i]: age_count += 1
    table_boxplot(labels_by_ex_id, datasets_by_ex_id, x_label='size median')

