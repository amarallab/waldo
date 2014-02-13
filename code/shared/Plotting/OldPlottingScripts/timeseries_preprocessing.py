import os
import sys
code_directory =  os.path.dirname(os.path.realpath(__file__)) + '/../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)
from Shared.Code.Database.mongo_retrieve import get_data_for_blob_id
from Shared.Code.Database.mongo_retrieve import get_data_for_ex_id
import numpy as np

def pull_timeseries_datasets_for_ex_id(ex_id, data_type, blob_filter={}):
    blob_ids, blob_data_sets = get_data_for_ex_id(ex_id, data_type, blob_filter, return_as_list=True)
    return blob_ids, blob_data_sets


def pull_timeseries_dataset_for_blob_id(blob_id, data_type, blob_filter={}):
    blob_data_set = get_data_for_blob_id(blob_id, data_type, blob_filter, return_as_list=True)
    return blob_data_set
    
def average_across_blobs(blob_data_sets, bin_times=True):
    '''
    for every timestep it calculates the mean and makes a new timeseries.
    TODO: expand this for other stats across the timeseries
    '''

    mean_line = []
    # do not rebin data, use existing timepoints

    points_in_bins = {}
    for blob_timeseries in blob_data_sets:
        #print blob_timeseries[0]
        for t, s in blob_timeseries:

            # if bin size = -1, just use given times as the bins.
            # this doesn't work if there are slight differences in times for blobs.
            if bin_times: bin_key = round(t,1)
            else: bin_key = t
            if bin_key not in points_in_bins: points_in_bins[bin_key] = []
            points_in_bins[bin_key].append(s)

    for t in points_in_bins:
        mean_line.append((t, np.mean(points_in_bins[t])))


    return sorted(mean_line)
