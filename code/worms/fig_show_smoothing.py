# standard imports
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
import matplotlib.mlab as mlab

import database.mongo_support_functions as mongo
from database.mongo_retrieve import mongo_query
from database.mongo_retrieve import pull_data_type_for_blob
from database.mongo_retrieve import timedict_to_list

from settings.local import LOGISTICS as logistics_settings
from settings.local import MONGO as mongo_settings
from experiment_index import Experiment_Attribute_Index

import flag_timepoints

DATA_DIR = logistics_settings['filesystem_data']
SAVE_DIR = './../Results/Smoothing/'

def choose_ex_id(key='purpose', value='N2_aging'):
    ei = Experiment_Attribute_Index()
    return ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value)

def plot_filter_timeseries(blob_id, **kwargs):
    savename = '{d}bid{bid}.png'.format(d=SAVE_DIR, bid=blob_id)
    print savename
    width_document = pull_data_type_for_blob(blob_id, 'width50', **kwargs)
    width_timedict = width_document['data']
    length_timedict = pull_data_type_for_blob(blob_id, 'length', **kwargs)['data']

    times, widths = timedict_to_list(width_timedict)
    _, lengths = timedict_to_list(length_timedict)

    smoothing_graphs(times, widths, lengths, savename=savename)

def fit_gaussian(x, num_bins=200):
    print len(x)
    n, bins = np.histogram(x, num_bins, normed=1)
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    mu, sigma = flag_timepoints.fit_gaussian(x)
    y = mlab.normpdf(bincenters, mu, sigma)
    return bincenters, y

def smoothing_graphs(times, widths, lengths, num_bins=30, savename=''):

    plt.figure(1)

    if savename:
        plt.savefig(savename)
    else:
        plt.show()
    plt.clf()


def main():
    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['blob_collection'])
    try:
        ex_ids = choose_ex_id()
        for ex_id in ex_ids[5:6]:
            print ex_id
            blobs = [e['blob_id'] for e in mongo_query({'ex_id': ex_id, 'data_type': 'smoothed_spine'}, {'blob_id':1})]
            for blob_id in blobs:
                print blob_id
                plot_filter_timeseries(blob_id, mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)


def plot_one_blob():
    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['blob_collection'])

    try:
        #blob_id = '20130318_153742_02796'
        blob_id = '20130320_102312_10828'
        blob_id = '20130320_102312_19189'

        plot_filter_timeseries(blob_id, mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)

if __name__ == '__main__':
    #plot_one_blob()
    main()
