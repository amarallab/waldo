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
from wio.file_manager import get_data
from importing.flags_and_breaks import consolidate_flags
from filtering.filter_utilities import filter_stat_timedict as fst
from show_measure import quickplot_stat2

def compute_size_measures(blob_id, **kwargs):
    '''
    :param blob_id:
    '''
    times_f, all_flags = get_data(blob_id, data_type='flags', **kwargs)
    times_s, sizes = get_data(blob_id, data_type='size_raw', **kwargs)
    flags = consolidate_flags(all_flags)
    pixels_per_mm = float(flag_data_entry.get('pixels-per-mm', 1.0))
    # making robust to depreciated notation that should no longer be in database
    #if pixels_per_mm == 1.0:
    #    pixels_per_mm = float(flag_data_entry.get('pixels_per_mm', 1.0))
    # pull size data, remove flagged timepoints, rescale, and return
    unflagged_timeseries = [(t, s) for (t, s, f) in izip(times_s, sizes, flags) if f]
    times, sizes = zip(*unflagged_timeseries)
    rescaled_sizes = np.array(sizes) / (pixels_per_mm**2)
    return times, rescaled_sizes


def compute_width_measures(blob_id, metric='all', **kwargs):
    '''

    :param blob_id:
    :param metric:
    '''
    assert metric in ['all', 'width_mm', 'width_bl']
    times_f, all_flags, doc = get_data(blob_id, data_type='flags', **kwargs)
    times, widths, _ get_data(blob_id, data_type='width50', **kwargs)
    flags = consolidate_flags(all_flags)


    pixels_per_mm = float(doc.get('pixels-per-mm', 1.0))
    pixels_per_bl = float(doc.get('pixels-per-body-length', 1.0))
    # making robust to depreciated notation that should no longer be in database
    #if pixels_per_mm == 1.0:
    #    pixels_per_mm = float(flag_data_entry.get('pixels_per_mm', 1.0))
    #if pixels_per_bl == 1.0:
    #    pixels_per_bl = float(flag_data_entry.get('pixels_per_body_length', 1.0))
    if pixels_per_bl == 1.0:
        pixels_per_bl = float(flag_data_entry.get('midline-median', 1.0))

    unflagged_timeseries = [(t, w) for (t, w, f) in izip(times, widths, flags) if f]
    times, widths = zip(*unflagged_timeseries)
    #unflagged_width_timedict = keep_unflagged_timepoints(flag_timedict, width_timedict)
    #width_mm_timedict = rescale_timedict(unflagged_width_timedict, pixels_per_mm)
    #width_bl_timedict = rescale_timedict(unflagged_width_timedict, pixels_per_bl)
    if metric =='width_mm':
        return times, np.array(widths) / (pixels_per_mm)
    if metric =='width_mm':
        return times, np.array(widths) / (pixels_per_bl)
    return {'width_mm': np.array(widths) / (pixels_per_mm),
            'width_bl': np.array(widths) / (pixels_per_bl)}

if __name__ == "__main__":
    #blob_id = '20121119_162934_07337'
    blob_id = '20121118_165046_01818'
    blob_id = '20130324_115435_04452'
    #blob_id = '00000000_000001_00002'
    metric = 'width_bl'
    #metric = 'width_mm'
    #metric = 'size_mm2'
    stat_timedict1 = compute_basic_measures(blob_id, metric=metric)#, smooth=False)
    stat_timedict2 = compute_basic_measures(blob_id, metric=metric)#, smooth=True)
    #print len(stat_timedict1), len(stat_timedict2)
    #quickplot_stat2(stat_timedict1, stat_timedict2, 'raw', 'smoothed')
