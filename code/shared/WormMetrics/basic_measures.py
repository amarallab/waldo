'''
Author: Peter Winter + Andrea L.
Date: jan 11, 2013
Description:
'''
# standard imports
import os
import sys
import numpy as np

# set paths
code_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)

# nonstandard imports
import wio.file_manager
from wio.file_manager import get_data
from importing.flags_and_breaks import consolidate_flags
from filtering.filter_utilities import filter_stat_timedict as fst
from show_measure import quickplot_stat2
'''
def keep_unflagged_timepoints(flags, data):
    unflagged_timedict = {}
    missmatched_key_count = 0
    for t in sorted(flag_timedict):

        #assert t in data, str(t) + ' not in timedict'
        if unicode(t) in data:
            if flag_timedict[t]:
                unflagged_timedict[t] = data[t]
        else:
            missmatched_key_count += 1
            if missmatched_key_count < 10:
                print str(t) + ' not in timedict', missmatched_key_count


    if missmatched_key_count > 0 :
        print 'There are this many missmatched keys', missmatched_key_count
        print 'Flag timedict has', len(flag_timedict) 
        print 'Other timedict has', len(data)
        for i in sorted(flag_timedict.keys())[:10]:
            print 'a', i, len(i)
        for i in sorted(k for k in data.keys() if len(k) == 7)[:10]:
            print 'b', i, len(i)


    return unflagged_timedict

def rescale_timedict(timedict, scaling_factor=1):
    rescaled_timedict = {}
    for t in sorted(timedict):
        rescaled_timedict[t] = timedict[t]/scaling_factor
    return rescaled_timedict
'''

def compute_size_measures(blob_id, smooth=True, **kwargs):
    '''

    :param blob_id:
    :param metric:
    :param smoothed:
    '''


    times_f, all_flags = get_data(blob_id, data_type='flags', **kwargs)
    metadata = get_data(blob_id, data_type='metadata', split_time_and_data=False)
    times_s, sizes = get_data(blob_id, data_type='size_raw', **kwargs)

    flags = consolidate_flags(all_flags)
    pixels_per_mm = float(flag_data_entry.get('pixels-per-mm', 1.0))
    pixels_per_bl = float(flag_data_entry.get('pixels-per-body-length', 1.0))
    # making robust to depreciated notation that should no longer be in database
    if pixels_per_mm == 1.0:
        pixels_per_mm = float(flag_data_entry.get('pixels_per_mm', 1.0))
    if pixels_per_bl == 1.0:
        pixels_per_bl = float(flag_data_entry.get('pixels_per_body_length', 1.0))
    if pixels_per_bl == 1.0:
        pixels_per_bl = float(flag_data_entry.get('midline-median', 1.0))


    # pull size data, remove flagged timepoints, rescale, and return
    unflagged_sizes = [s for (s,f) in izip(sizes, flags) if f]
    rescaled_sizes = np.array(unflagged_sizes) / (pixels_per_mm**2)
    if smooth:
        return fst(rescaled_sizedict)
    else:
        return rescaled_sizedict


def compute_width_measures(blob_id, metric='all', source_data_entry=None, flag_data_entry=None, smooth=True, **kwargs):
    '''

    :param blob_id:
    :param metric:
    :param smoothed:
    '''
    assert metric in ['all', 'width_mm', 'width_bl']

    if flag_data_entry:
        assert flag_data_entry['data_type'] == 'flags', 'Error: wrong type of flag data entry provided'
    else:
        flag_data_entry = pull_data_type_for_blob(blob_id, 'flags', **kwargs)

    all_flag_dicts = flag_data_entry['data']
    flag_timedict = consolidate_flags(all_flag_dicts)
    pixels_per_mm = float(flag_data_entry.get('pixels-per-mm', 1.0))
    pixels_per_bl = float(flag_data_entry.get('pixels-per-body-length', 1.0))

    # making robust to depreciated notation that should no longer be in database
    if pixels_per_mm == 1.0:
        pixels_per_mm = float(flag_data_entry.get('pixels_per_mm', 1.0))
    if pixels_per_bl == 1.0:
        pixels_per_bl = float(flag_data_entry.get('pixels_per_body_length', 1.0))
    if pixels_per_bl == 1.0:
        pixels_per_bl = float(flag_data_entry.get('midline-median', 1.0))


    datatype = 'width50'
    if source_data_entry:
        assert source_data_entry['data_type'] == datatype, 'Error: wrong type of data entry provided'
    else:
        source_data_entry = pull_data_type_for_blob(blob_id, data_type=datatype, **kwargs)

    width_timedict = source_data_entry['data']
    unflagged_width_timedict = keep_unflagged_timepoints(flag_timedict, width_timedict)
    width_mm_timedict = rescale_timedict(unflagged_width_timedict, pixels_per_mm)
    width_bl_timedict = rescale_timedict(unflagged_width_timedict, pixels_per_bl)

    if metric =='width_mm':
        if smooth:
            return fst(width_mm_timedict)
        else: return width_mm_timedict
    if metric == 'width_bl':
        if smooth:
            return fst(width_bl_timedict)
        else: return width_bl_timedict
    return {'width_mm': fst(width_mm_timedict),
            'width_bl': fst(width_bl_timedict)}


if __name__ == "__main__":
    #blob_id = '20121119_162934_07337'
    blob_id = '20121118_165046_01818'
    blob_id = '20130324_115435_04452'
    #blob_id = '00000000_000001_00002'
    metric = 'width_bl'
    #metric = 'width_mm'
    #metric = 'size_mm2'
    stat_timedict1 = compute_basic_measures(blob_id, metric=metric, smooth=False)
    stat_timedict2 = compute_basic_measures(blob_id, metric=metric, smooth=True)
    print len(stat_timedict1), len(stat_timedict2)
    quickplot_stat2(stat_timedict1, stat_timedict2, 'raw', 'smoothed')
