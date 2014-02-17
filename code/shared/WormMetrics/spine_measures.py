'''
Author: Peter Winter + Andrea L.
Date: jan 11, 2013
Description:
'''

# standard imports
import os
import sys
import math
import numpy as np
import pylab as pl

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIRECTORY = os.path.abspath(HERE + '/../')
CODE_DIRECTORY = os.path.abspath(SHARED_DIRECTORY + '/../')
sys.path.append(CODE_DIRECTORY)
sys.path.append(SHARED_DIRECTORY)

# nonstandard imports
#from database.mongo_retrieve import pull_data_type_for_blob
#from database.mongo_retrieve import timedict_to_list
from wio.file_manager import get_data
from GeometricCalculations import compute_displacement_along_curve
from GeometricCalculations.distance import euclidean
from filtering.filter_utilities import filter_stat_timedict as fst

def make_it_str(t):
    return str('%.3f' % t).replace('.', '?')

def compute_curvature_from_three_points(p, q, t):
    """
        Solution from: http://mathworld.wolfram.com/SSSTheorem.html
        p,q and t are tuples (x,y)
    """
    a = euclidean(p, q)
    b = euclidean(t, q)
    c = euclidean(p, t)

    if a == 0 or b == 0 or c == 0:
        return 0.

    s = 0.5 * (a + b + c)
    # preventing round off error from breaking code
    if (s * (s - a) * (s - b) * (s - c)) < 0:
        K = 0.
    else:
        K = math.sqrt(s * (s - a) * (s - b) * (s - c))

    return K / (a * b * c) * 4.


def compute_curvature(spine, points, scaling_factor=1):
    if points == 'all':
        points = range(len(spine))

    curvatures = []
    for i in points:
        if 1 < i < len(spine) - 1:
            curvature = compute_curvature_from_three_points(spine[i - 1], spine[i], spine[i + 1])
            curvatures.append(curvature / scaling_factor)

    return np.mean(curvatures)


def compute_all_curvatures(times, spine_list, points, scaling_factor=1, verbose=False):
    """
        same arguments as other functions in this module
    """

    curvature_timedict = {}

    for i in range(len(times)):
        if len(spine_list[i]) > 0:
            curvature_timedict[make_it_str(times[i])] = compute_curvature(spine_list[i], points, scaling_factor)
        else:
            curvature_timedict[make_it_str(times[i])] = 'skipped'

    if verbose:
        print points, 'points'
        for k in sorted(curvature_timedict.keys()[:10]):
            print k, curvature_timedict[k]
        print '------------------------------'

    return curvature_timedict

def compute_length(spine):
    na_values = ['', -1, 'NA', 'NaN', None, []]
    if len(spine) == 0 or spine in na_values:
        return 'NA'
    dx, dy = map(np.diff, map(np.array, zip(*spine)))
    return np.sqrt(dx**2 + dy**2).sum()

def compute_speed_along_spine(spine1, spine2, t_plus_dt, t, perpendicular, points, median_length, show_plot):
    '''
        this function takes a spine
        list of [[x1,y1], [x2,y2], ...]
        and returns the average speed along (or perpendicular to) the spine
    '''
    dt = t_plus_dt - t
    assert dt > 0, 'times are not sorted!'
    assert len(spine1) > 0, 'spine1 is empty'
    assert len(spine1) == len(spine2), 'spines of different lengths'
    if points == 'all':
        points = range(len(spine1))

    displacement = compute_displacement_along_curve(spine1, spine2, perpendicular=perpendicular, points=points)
    speed = displacement / len(points) / dt / float(median_length)

    if show_plot:
        ax = pl.subplot(111)
        xs, ys = zip(*spine1)
        pl.plot(xs, ys)
        pl.plot([xs[0]], [ys[0]], marker='o', color='red')
        pl.xlim(int(min(xs) - 10), int(max(xs) + 10))
        pl.ylim(int(min(ys) - 10), int(max(ys) + 10))

        ax.set_title(
            str(round(displacement, 2)) + ' ' + str(round(speed, 2)) + ' ' + str(round(t, 2)) + ' ' + str(len(points)))
        pl.draw()
        pl.clf()

    return speed

def compute_speeds_with_options(times, spine_list, perpendicular, points, median_length=1, verbose=False,
                                show_plot=False):
    """
        times and spine_list are the input data (see below)
        if perpendicular is false we compute the speed along the spine
        points is to consider only some points across the spine
        should be a list of indices like range(0,len(spine_list[0])/3)
        if instead is a string: 'all', it will be set equal to range(0, len(spine_list[0]))
        if median_length is not specified, it will compute speed in terms of pixels
    """

    speed_timedict_options = {}
    if show_plot:
        pl.ion()
    for i in range(len(times)):
        if i + 1 < len(spine_list) and len(spine_list[i]) > 0 and len(spine_list[i + 1]) > 0:

            speed = compute_speed_along_spine(spine_list[i], spine_list[i + 1],
                                              times[i + 1], times[i], perpendicular, points, median_length, show_plot)
            speed_timedict_options[make_it_str(times[i])] = speed
        else:
            speed_timedict_options[make_it_str(times[i])] = 'skipped'

    if verbose:
        print points, 'points: median_length', median_length
        for k in sorted(speed_timedict_options.keys()[:10]):
            print k, speed_timedict_options[k]
        print '------------------------------'
    return speed_timedict_options


def compute_spine_measures(blob_id, metric='all', smooth=False, source_data_entry=None, **kwargs):
    '''
        pulls smoothed_spine entry for blob_id, calculates length, speeds, and curvatures and
        inserts them into the 'Results' mongodb collection
    :param metric:
    :param blob_id:
    '''
    '''
    datatype='smoothed_spine'

    if source_data_entry:
        assert source_data_entry['data_type'] == datatype, 'Error: wrong type of data entry provided'
    if not source_data_entry:
        source_data_entry = pull_data_type_for_blob(blob_id, data_type=datatype, **kwargs)

    # pull source data from database
    spine_timedict = source_data_entry['data']

    assert len(spine_timedict) > 0, 'Warning: spine_list is empty\n' + str(source_data_entry)
    times, spine_list = timedict_to_list(spine_timedict)
    spine_len = len(spine_list[0])
    assert len(times) == len(spine_list), 'times and spine_list have different lengths in compute_speeds_for_blob_id'
    '''
    times, spines = get_data(blob_id, data_type='smoothed_spine')
    spine_len = max([len(s) for s in spines])
    # compute scaling factors for body length and
    lengths = [compute_length(s) for s in spines]
    median_length_pixels = np.median(lengths)

    # find scaling factor for this worm or leave it at 1
    pixels_per_mm = float(source_data_entry.get('pixels-per-mm', 1.0))
    # making robust to depreciated notation that should no longer be in database
    if pixels_per_mm == 1.0:
        pixels_per_mm = float(source_data_entry.get('pixels_per_mm', 1.0))
    if pixels_per_mm == 1.0:
        print 'Spine Measures Warning: could not find pixels-per-mm for {ID}'.format(ID=blob_id)

    # make a big list of all combinations of speed calculations we would like measured
    #((data_type name for db), (options for compute_speeds_with_options), (description for db))


    head = range(0, spine_len / 3)
    midbody=  range(spine_len / 3, 2 * spine_len / 3)
    tail = range(2 * spine_len / 3, spine_len)
    measures = {'smooth_length': (None, None),
                # speed along
                'speed_along': (compute_speeds_with_options, (False, 'all', pixels_per_mm)),
                'speed_along_head': (compute_speeds_with_options, (False, head, pixels_per_mm)),
                'speed_along_mid': (compute_speeds_with_options, (False, midbody, pixels_per_mm)),
                'speed_along_tail': (compute_speeds_with_options, (False, tail, pixels_per_mm)),
                # speed along body length
                'speed_along_bl': (compute_speeds_with_options, (False, 'all', median_length)),
                'speed_along_head_bl': (compute_speeds_with_options, (False, head, median_length)),
                'speed_along_mid_bl': (compute_speeds_with_options, (False, midbody, median_length)),
                'speed_along_tail_bl': (compute_speeds_with_options, (False, tail, median_length)),
                # computing perpendicular speeds (pixels)
                'speed_perp': (compute_speeds_with_options, (True, 'all', pixels_per_mm)),
                'speed_perp_head': (compute_speeds_with_options, (True, head, pixels_per_mm)),
                'speed_perp_mid': (compute_speeds_with_options, (True, midbody, pixels_per_mm)),
                'speed_perp_tail': (compute_speeds_with_options, (True, tail, pixels_per_mm)),
                # speed along body length
                'speed_perp_bl': (compute_speeds_with_options, (True, 'all', median_length)),
                'speed_perp_head_bl': (compute_speeds_with_options, (True, head, median_length)),
                'speed_perp_mid_bl': (compute_speeds_with_options, (True, midbody, median_length)),
                'speed_perp_tail_bl': (compute_speeds_with_options, (True, tail, median_length)),
                # curvature in mm
                'curvature_all': (compute_all_curvatures, ('all', pixels_per_mm)),
                'curvature_head': (compute_all_curvatures, (head, pixels_per_mm)),
                'curvature_mid': (compute_all_curvatures, (midbody, pixels_per_mm)),
                'curvature_tail': (compute_all_curvatures, (tail, pixels_per_mm)),
                # curvature in bl
                'curvature_all_bl': (compute_all_curvatures, ('all', median_length)),
                'curvature_head_bl': (compute_all_curvatures, (head, median_length)),
                'curvature_mid_bl': (compute_all_curvatures, (midbody, median_length)),
                'curvature_tail_bl': (compute_all_curvatures, (tail, median_length)),
                }

    assert (metric in measures) or (metric == 'all'), 'invalid metric'

    def pull_measure(measure_name):
        assert measure_name in measures
        call_function, options = measures[measure_name]

        # just for length
        if call_function == None:
            return length_timedict

        # for all speed calculations
        elif call_function == compute_speeds_with_options:
            (perpendicular, points, scaling_factor) = options
            return compute_speeds_with_options(times, spine_list, perpendicular, points, scaling_factor)

        # for all curvature calculations
        elif call_function == compute_all_curvatures:
            (points, scaling_factor) = options
            return compute_all_curvatures(times, spine_list, points, scaling_factor=scaling_factor)
        else:
            print measure_name, 'is messed up'
    body_sections = {'': 'all', '_head': head, '_mid': midbody, '_tail': tail}
    scaling_factors = {'': pixels_per_mm, '_bl': median_length}


    if metric == 'all':
        all_datasets = {}
        m = 'speed'
        for speed_dir, is_perp in [('_along', False), ('_perp', True)]:
            for bsec, point_range in body_sections.iteritems():
                # calculate the speed type for that body section.
                speed_dict = compute_speeds_with_options(times=times, spine_list=spine_list,
                                                         perpendicular=is_perp, points=point_range)
                for sf_type, sf in scaling_factors.iteritems():
                    name = m + speed_dir + bsec + sf_type
                    all_datasets[name] = rescale_timedict(speed_dict, scaling_factor=float(sf))

        m = 'curvature'
        for bsec, point_range in body_sections.iteritems():
            # calculate the speed type for that body section.
            curve_dict = compute_all_curvatures(times=times, spine_list=spine_list, points=point_range)
            for sf_type, sf in scaling_factors.iteritems():
                name = m + bsec + sf_type
                all_datasets[name] = rescale_timedict(curve_dict, scaling_factor=float(sf))

        return all_datasets
    elif smooth:
        return fst(pull_measure(metric))
    else:
        return pull_measure(metric)

def rescale_timedict(timedict, scaling_factor=1):
    rescaled_timedict = {}
    for t in timedict:
        value = timedict[t]
        if type(value) in [int, float]:
            rescaled_timedict[t] = value / scaling_factor
        else:
            rescaled_timedict[t] = value
    return rescaled_timedict


if __name__ == "__main__":
    #blob_id = '20121119_162934_07337'
    blob_id = '20130324_115435_04452'
    blob_id = '00000000_000001_00003'
    blob_id = '20130319_150235_01070'
    import time
    '''
    import time
    start = time.time()
    compute_spine_measures2(blob_id=blob_id, metric='all')
    dur1 = time.time() - start
    print 'time1', dur1
    '''
    start = time.time()
    compute_spine_measures(blob_id=blob_id, metric='all')
    dur2 = time.time() - start
    print 'time2', dur2

    #print dur2/dur1

    #stat_timedict = compute_spine_measures(blob_id, metric='speed_along_head', smooth=False)
    #stat_timedict = compute_spine_measures(blob_id, metric='speed_along', smooth=False)
    #stat_timedict = compute_spine_measures(blob_id, metric='curvature_all', smooth=False)
    #stat_timedict = compute_spine_measures(blob_id, metric='speed_perp', smooth=False, datatype='smoothed_spine')
    #stat_timedict = compute_spine_measures(blob_id, metric='smooth_length', smooth=False)
    #quickplot_stat(stat_timedict)
    '''
    metric = 'speed_perp'
    metric = 'speed_along'
    stat_timedict1= compute_spine_measures(blob_id, metric=metric, smooth=False, datatype='smoothed_spine')
    stat_timedict2= compute_spine_measures(blob_id, metric=metric, smooth=False, datatype='smoother_spine')
    quickplot_stat2(stat_timedict1, stat_timedict2, 'single', 'repeated iterations')
    '''
    #compare_spine_measure_functions(blob_id)
