import os
import sys
code_directory =  os.path.dirname(os.path.realpath(__file__)) + '/../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)
import numpy as np
from Shared.Code.Plotting.OldPlottingScripts.timeseries_preprocessing import *
from pylab import *
import matplotlib.pyplot as plt
from Shared.Code.Database.mongo_retrieve import mongo_query
from Shared.Code.Database.mongo_retrieve import pull_data_type_for_blob
#from Database.mongo_search import pick_random_blob_id
from Shared.Code.Database.mongo_retrieve import timedict_to_list

def plot_x_y_and_hist(x,y, title):
    '''
    speeds is a list of speed data. each entry contains
    '''
    # definitions for the axes
    left, width = 0.15, 0.65
    bottom, height = 0.15, 0.75
    bottom_h = left_h = left+width+0.02
    rect_timeseries = [left, bottom, width, height]
    rect_histy = [left_h, bottom, 0.1, height]

    # plot the timeseries
    plt.figure(1, figsize=(12,4))
    axTimeseries = plt.axes(rect_timeseries)
    plt.plot(x, y) 
    # now determine nice limits by hand:
    binwidth = np.std(y)/100
    xymax = np.max( [np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth
    axTimeseries.xaxis.set_label_text('time (s)')
    axTimeseries.yaxis.set_label_text(title)

    #bins = np.arange(-lim, lim + binwidth, binwidth)
    
    #hist, bins = np.histogram(x,bins = 200)
    axHisty = plt.axes(rect_histy)
    num_bins = 200
    n, bin_edges =  np.histogram(y, num_bins, normed=True)
    bincenters = [0.5*(bin_edges[i+1]+bin_edges[i]) for i in range(len(n))]

    #axHisty.hist(hist, bins=bincenters, orientation='horizontal', alpha=.7)
    plt.plot(n ,bincenters)
    plt.fill_between(n, [0 for i in n], bincenters, facecolor='blue', alpha=0.5)
    #axHisty.hist(y, bins=bins, orientation='horizontal')
    axHisty.set_ylim( axTimeseries.get_ylim())
    axHisty.set_xlim([0,max(n)])
    axHisty.axes.get_yaxis().set_visible(False)
    axHisty.axes.get_xaxis().set_visible(False)
    #for tick in axHisty.axes.get_xticklines(): tick.set_visible(False)
    #for tick in axHisty.axes.get_yticklines(): tick.set_visible(False)
        

    #plt.show()
    plt.savefig(title+'.png')
    plt.clf()

def plot_x_y_speed(speed_datasets, x_datasets, y_datasets, labels):
    '''
    speeds is a list of speed data. each entry contains
    '''
    figure()
    subplot(3, 1, 1)
    for speed_dset in speed_datasets:
        times, speeds = zip(*speed_dset)
        plot(times, speeds)


    subplot(3, 1, 2)
    for label, x_dset in zip(labels, x_datasets):
        times, x = zip(*x_dset)        
        plot(times, x)

    subplot(3, 1, 3)
    for label, y_dset in zip(labels, y_datasets):
        times, y = zip(*y_dset)
        plot(times, y)

    show()

def graph_position_timeseries(t1, x1, t2, x2):
    figure()
    plot(t1,x1, color='b', marker='x')
    plot(t2,x2 , color='r', marker='o')
    show()

def get_metadata(ex_id, **kwargs):
    date,time_stamp = ex_id.split('_')
    entries = mongo_query({'date':date,'time_stamp':time_stamp,'data_type':'metadata'},
                          {'name':1}, **kwargs)
    e = [entry['name'] for entry in entries]
    return e[0]

if __name__ == "__main__":
    blob_id ='20120914_172813_01708'
    datatypes = ['speed_ratio', 
                 'curvature_all',
                 'curvature_head',
                 'curvature_tail',
                 'speed_perp_tail_bl',
                 'speed_perp_head_bl',
                 'speed_perp_bl',
                 'speed_perp_tail',
                 'speed_perp_head',
                 'speed_perp',
                 'speed_along_bl',
                 'speed_along_head_bl',
                 'speed_along_tail_bl',
                 'speed_along_tail',
                 'speed_along_head',
                 'speed_along',
                 'smooth_length']



    #data_type = datatypes[4]
    #data_type = 'curvature_head'
    #i = 2
    #for data_type in datatypes[i:i+1]:
    for data_type in datatypes:
        x, y = [], []
        data_entry = pull_data_type_for_blob(blob_id, data_type)
        time, data = timedict_to_list(data_entry['data'])
        for t, d in zip(time,data):
            if d != 'skipped':
                x.append(t)
                y.append(d)
            else:
                print 'skipping', t, d
    
        plot_x_y_and_hist(x,y, title=data_type)
         
