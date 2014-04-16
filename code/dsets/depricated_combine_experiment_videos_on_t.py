#!/usr/bin/env python

'''
Filename: plot_plate_speeds.py
Description:  
'''
__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
import glob

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_directory)

# nonstandard imports
import Shared.Code.Database.mongo_support_functions as mongo
from Shared.Code.Database.mongo_retrieve import mongo_query, pull_data_type_for_blob, timedict_to_list
from Shared.Code.Settings.data_settings import mongo_settings
from Import.Code.experiment_index import Experiment_Attribute_Index
from compute_N_for_plate import compute_N

DATA_DIR = os.path.abspath('./../Data/')
RESULTS_DIR = os.path.abspath('./../Results/')
print RESULTS_DIR

def choose_ex_id(key='purpose', value='N2_aging'):
    ei = Experiment_Attribute_Index()
    return ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value)


def plot_speed_of_plate(ex_id, save_dir=RESULTS_DIR, **kwargs):
    """
    Generates two timeseries graphs for a plate that are stacked vertically for contrast.

    subplot 1: The number of worms being tracked at any given time during the recording ploted in three ways.
    subplot 2: The average centroid speed of all the objects/worms tracked at any given time. 

    Arguments:
    - `ex_id`: the ID of the plate to make a figure for.
    - `save_dir`: the directory in which the figures should be saved.
    - `**kwargs`: to pass along the mongo_client object for queries
    """

    save_name = '{dir}/Speed/{ex_id}.png'.format(dir=save_dir, ex_id=ex_id)
    print save_name

    # grab necessary data from plate collection in database
    summary_doc = mongo_query({'ex_id': ex_id, 'type':'stage0'}, find_one=True, col='plate_collection', **kwargs)
    summary_data = summary_doc['data']
    stage1_doc = mongo_query({'ex_id': ex_id, 'type':'stage1'}, find_one=True, col='plate_collection', **kwargs)
    stage1_data = stage1_doc['data']
    times, all_N, good_N = compute_N(ex_id)
    
    # plot toggles
    range_x = [0, 3600]
    obj_color = 'green'
    perst_color = 'blue'
    stage1_color = 'red'

    fig = plt.figure()
    # Subplot1: Number of objects over time
    ax = plt.subplot(2,1,1)
    ax.plot(summary_data['time'], summary_data['N'], color=obj_color, alpha=0.5) #, label='objects')
    ax.plot(summary_data['time'], summary_data['N-persist'], color=perst_color, alpha=0.5) #, label='persisting objects')
    ax.plot(stage1_data['time'], stage1_data['N'], color=stage1_color, alpha=0.5) #, label='stage1 worms')
    ax.plot(times, all_N, color='yellow')
    ax.plot(times, good_N, color='black')
    ax.legend(loc='lower right')
    ax.set_xlim(range_x)
    ax.set_ylabel('N')

    # Subplot2: speed of objects over time
    ax = plt.subplot(2,1,2)
    # include this line so that objects will be in legend on second subplot
    ax.plot([], [], color=obj_color, label='objects')
    ax.plot(summary_data['time'], summary_data['px-per-s'], color=perst_color, alpha=0.5, label='persisting objects')
    ax.plot(stage1_data['time'], stage1_data['median'], color=stage1_color, alpha=0.5, label='stage1 worms')
    ax.set_xlim(range_x)
    ax.legend(loc='upper right')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('pixels per second (s)')

    #plt.show()
    plt.savefig(save_name)
    plt.clf()

def read_plate_timeseries_file(filename, return_type='all'):
    assert return_type in ['means', 'all', 'N', 'top50']
    times = []
    data = []
    with open(filename, 'r') as f:
        for line in f:
            cols = line.split(',')
            t = float(cols[0])
            times.append(t)
            if return_type == 'means':
                data.append(np.mean(map(float, cols[1:])))
            elif return_type == 'N':
                data.append(len(cols[1:]))
            elif return_type == 'top50':
                top50 = sorted(map(float,cols[1:]))
                data.append(np.mean(top50[-50:]))
            elif return_type=='all': 
                data.append(map(float, cols[1:]))
    return times, data

def count_ex_ids_present(data_type='centroid_speed'):
    search_path = '{path}/{dt}/*_*.txt'.format(path=DATA_DIR, dt=data_type)
    data_files = glob.glob(search_path)

    ex_id_to_file = {}
    for f in data_files:
        ex_id_to_file[f.split('/')[-1].split('.txt')[0]] = f

    ei = Experiment_Attribute_Index()
    ex_ids_to_plot = ei.return_ex_ids_with_attribute(key_attribute='purpose', attribute_value='N2_aging')
    ages = ei.return_attribute_for_ex_ids(ex_ids=ex_ids_to_plot, attribute='age')

    ex_ids_by_age = {}
    for (a, e) in izip(ages, ex_ids_to_plot):
        age = int(a[1:])
        if age not in ex_ids_by_age:
            ex_ids_by_age[age] = []
        ex_ids_by_age[age].append(e)

    print len(ex_id_to_file), 'files'
    print len(ex_ids_to_plot), 'to plot'
    for age in range(12):
        ex_ids = ex_ids_by_age.get(age, [])
        N_ex_ids = len(ex_ids)
        ex_ids_present = [e for e in ex_ids if e in ex_id_to_file]
        N_present = len(ex_ids_present)
        print age, N_present, N_ex_ids

def combine_all_ages(data_type='centroid_speed'):
    """
        
    Arguments:
    - `ex_ids`:
    - `data_type`:
    """
    
    search_path = '{path}/{dt}/*_*.txt'.format(path=DATA_DIR, dt=data_type)
    data_files = glob.glob(search_path)

    ex_id_to_file = {}
    for f in data_files:
        ex_id_to_file[f.split('/')[-1].split('.txt')[0]] = f

    ei = Experiment_Attribute_Index()
    ex_ids_to_plot = ei.return_ex_ids_with_attribute(key_attribute='purpose', attribute_value='N2_aging')
    ages = ei.return_attribute_for_ex_ids(ex_ids=ex_ids_to_plot, attribute='age')

    ex_ids_by_age = {}
    for (a, e) in izip(ages, ex_ids_to_plot):
        age = int(a[1:])
        if age not in ex_ids_by_age:
            ex_ids_by_age[age] = []
        ex_ids_by_age[age].append(e)

    for age in ex_ids_by_age:
        savename = '{path}/{dt}/age{a}_combined.txt'.format(path=DATA_DIR, dt=data_type, a=age)
        print savename
        ex_ids = ex_ids_by_age[age]
        data_by_timepoint = {}
        for ex_id in ex_ids:
            if ex_id in ex_id_to_file:
                data_file = ex_id_to_file[ex_id]
                times, data = read_plate_timeseries_file(data_file)
                for t, d in izip(times, data):
                    if t not in data_by_timepoint:
                        data_by_timepoint[t] = []
                    data_by_timepoint[t] += d

        with open(savename, 'w') as f:
            for t in sorted(data_by_timepoint):
                line = '{t},{d}'.format(t=t, d=','.join(map(str,data_by_timepoint[t])))
                f.write(line + '\n')

def plot_all_ages(data_type='centroid_speed', ages=range(1, 10)):
    search_path = '{path}/{dt}/age*_combined.txt'.format(path=DATA_DIR, dt=data_type)
    dfiles_by_age = {}

    labelsize = 15
    tick_settings = {'top': 'on', 'bottom': 'on', 'right': 'on', 'left': 'on',
                      'labelbottom': 'on', 'labeltop': 'off', 'labelright': 'off', 'labelleft': 'off',
                      'labelsize':15}
    
    for dfile in glob.glob(search_path):
        print dfile
        age = dfile.split('/age')[-1].split('_combined')[0]
        dfiles_by_age[int(age)] = dfile

    #settings = {'dtype':'N', 'ylim':[0,1000], 'ylabel':'N values per bin'}
    settings = {'dtype':'means', 'ylim':[0,0.2], 'ylabel':'speed (mm/s)'}
    #settings = {'dtype':'top50', 'ylim':[0,0.35], 'ylabel':'speed (mm/s)'}

    numplots = len(ages)
    for i, age in enumerate(ages):
        filename = dfiles_by_age.get(age, None)
        if not filename:
            print 'warning, no file found for age', age
            continue
        times, data = read_plate_timeseries_file(filename, return_type=settings['dtype'])
        times = [t/3600.0 for t in times]
        ax = plt.subplot(1, numplots, age)
        ax.set_ylim(settings['ylim'])
        ax.set_xlim([0.01, 1])
        ax.set_xticks([0.5, 1.0])
        ax.set_title(str(age), fontsize=labelsize)
        ax.plot(times, data, color='blue')
        ax.fill_between([0, 1], [max(data), max(data)], [min(data), min(data)],color='black', alpha=0.1)
        plt.tick_params(**tick_settings)
        # only the first graph should have labels on the bottom
        #tick_settings['labelbottom'] = 'off'
        if i == 0:
            ax.set_xlabel('time (h)', fontsize=labelsize)



    # add y axis on first and last section
    ax = plt.subplot(1, numplots, 1)
    ax.set_ylabel(settings['ylabel'], fontsize=labelsize)
    #ax.set_ylabel('number of points per bin')
    plt.tick_params(labelleft='on')
    ax = plt.subplot(1, numplots, numplots)
    #ax.ylabel.set_label_position('right')
    plt.tick_params(labelright='on')
    #ax.set_ylabel('speed (mm/s)')
    #plt.legend()
    plt.show()

def main(ex_ids):
    """
    runs plot_speed_of_plate() repeatidly for a list of ex_ids and maintaining the same mongo_connection.
    """
    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['plate_collection'])
    try:
        for ex_id in ex_ids:
            try:
                plot_speed_of_plate(ex_id, mongo_client=mongo_client)
            except Exception as e:
                print e
                print 'WARNING:', ex_id, 'not plotted'
            
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)

if __name__ == '__main__':
    #ex_ids = choose_ex_id()
    #main(ex_ids[:10])
    #combine_all_ages()
    plot_all_ages()
    #count_ex_ids_present()
