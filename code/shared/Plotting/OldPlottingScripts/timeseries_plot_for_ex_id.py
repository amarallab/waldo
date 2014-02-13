import os
import sys
code_directory =  os.path.dirname(os.path.realpath(__file__)) + '/../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)

from Shared.Code.Plotting.OldPlottingScripts.timeseries_preprocessing import *
from mongo_scripts.mongo_search import return_all_ex_ids
from pylab import *
from mongo_scripts.mongo_retrieve import mongo_query


def graph_ex_id_timeseries(mean_line, all_blob_datasets, savename, y_label=''):
    figure()
    for blob_timeseries in all_blob_datasets[:]:
        x, y = zip(*blob_timeseries)
        plot(x,y, color='b', alpha=.1) #, ls='1') #, alpha=0.2)#, label=str(leg))#, legend=legends[i])

    x, y = zip(*mean_line)
    plot(x,y,color='k') #, color='b', alpha=.1) #, ls='1') #, alpha=0.2)#, label=str(leg))#, legend=legends[i])
        
    #ylim([0,max_shown+1])
    #xlim([0,680])
    #ylim([0,1])
    xlabel('time seconds')
    #yticks([])
    ylabel(y_label)
    title(savename.split('/')[-1].rstrip('png'))
    print savename
    #show()
    savefig(savename)

def get_name_for_ex_id(ex_id):
    date,time_stamp = ex_id.split('_')
    entries = mongo_query({'date':date,'time_stamp':time_stamp,'data_type':'metadata'},
                          {'name':1})
    e = [entry['name'] for entry in entries]
    return e[0]

if __name__ == "__main__":
    all_ex_ids = return_all_ex_ids()
    blob_filter = {#'midline_median':{'$lt':1},
                   'age':{'$gt':'d5'},
                   'growth_medium':'liquid'}
    data_type = 'speed_filtered'
    #ex_id = all_ex_ids[0]
    for ex_id in all_ex_ids[:]:
        #ex_id = '20121126_185855'

        print ex_id
        try:
            blob_ids, blob_datasets = pull_timeseries_datasets_for_ex_id(ex_id, data_type, blob_filter)
            mean_line = average_across_blobs(blob_datasets)
            savename = '/home/codes/peterwinter/worm_movement/quick_results/ex_id_speed_timeseries_size/%s_%s_%s'%(str(ex_id),
                                                                                                                   str(get_name_for_ex_id(ex_id)),
                                                                                                                   'full.png')
            graph_ex_id_timeseries(mean_line, blob_datasets, savename, y_label='speed')
        except:
            print 'did not work'
