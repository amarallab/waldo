#!/usr/bin/env python
from __future__ import absolute_import, print_function

# standard library
#import os
import math

# third party
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.interpolate as interpolate
import scipy.signal as ss
import random

# project specific
from waldo.wio import paths
from waldo.wio.worm_writer import WormWriter
from waldo.wio.experiment import Experiment

STOCK_METHODS = [
    'boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett',
    'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann',
    'kaiser', 'gaussian', 'general_gaussian', 'slepian', 'chebwin'
]

def smooth(method, series, winlen, *params):
    """ generic smoothing function. smooths a series of values using
    the chosen method.

    params:
    method: (str)
        name of method to use for smoothing.
    series: (list or np.array)
        series of values that are to be smoothed
    *params: (*args)
        aditional arguments for chosen method.
    """
    try:
        winlen = int(winlen) // 2 * 2 + 1 # make it odd, rounding up
        #half_win = winlen // 2
        wintype = (method,) + tuple(int(x) for x in params)
        fir_win = ss.get_window(wintype, winlen)
    except ValueError:
        raise ValueError('Unrecognized smoothing type')

    b = fir_win / sum(fir_win)
    a = [1]
    return ss.lfilter(b, a, series)[winlen-1:]

def interpolate_to_1s(t, x):
    """ takes a timeseries and a value series.
    returns a timeseries where each time is a int and the value series has
    been linearly interpolated in between each point.
    """
    def one_second_timesteps(t):
        #print t[0]
        t0 = math.ceil(t[0])
        tN = math.floor(t[-1])
        return np.arange(t0, tN)
    eq_times = one_second_timesteps(t) #(t[0], t[-1], dt=dt, ndigits=0)
    interp_x = interpolate.interp1d(t, x, kind='linear')
    x_interp = interp_x(eq_times)
    return eq_times, x_interp

def calculate_speed(t, x, y):
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    #print len(dt), len(dx), len(dy)
    dist = np.sqrt(dx**2 + dy**2)
    s = dist/dt
    return s

def clean_speed(t, x, y):
    """ smooths and interpolates position of point and returns the speed.


    note: the boxcar window shaves off five points at beginning and end.
    the speed calculation shaves off one point at beginning

    params
    -----
    t, x, y

    returns
    -----
    ti: (np.array)
        series of times in 1 second intervals
    speed: (np.array)
        series of speeds
    """
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    window_size = 11
    half_win = window_size // 2
    ts = t[half_win:-half_win]
    xs = smooth('boxcar', x, window_size)
    ys = smooth('boxcar', y, window_size)
    #print 'smoothed', len(xs), len(ys)
    #print len(t), len(xs), len(ys), 'xs,ys'
    ti, xi = interpolate_to_1s(ts, xs)
    ti, yi = interpolate_to_1s(ts, ys)
    #print 'interp', len(ti), len(xi), len(yi)
    s2 = calculate_speed(ti, xi, yi)
    #print 'speeds', len(s2)
    return ti[1:], s2

def curate_data(node_df):
    """ accepts the
    """
    data=node_df.copy()
    data.reset_index(inplace=True)
    relevant_data = data[['frame', 'time', 'area', 'blob', 'contour_encode_len', 'contour_encoded']]
    x, y = zip(*data['centroid'])
    cx, cy = zip(*data['contour_start'])

    relevant_data['x'] = x
    relevant_data['y'] = y
    relevant_data['contour_start_x'] = cx
    relevant_data['contour_start_y'] = cy
    relevant_data = relevant_data[['frame', 'time','blob', 'x', 'y', 'area',
                                   'contour_start_x','contour_start_y', 'contour_encode_len', 'contour_encoded']]

    duplicates_present = len(data['frame']) > len(set(data['frame']))
    if duplicates_present:
        #print 'duplicates found in', data.iloc[0]['blob']
        duplicated_frames = relevant_data[relevant_data.duplicated('frame')]['frame']
        #print len(duplicated_frames), 'frames duplicated'
        new_rows = []
        for f in duplicated_frames:
            dup = relevant_data[relevant_data['frame'] == f]
            total_area = dup['area'].sum()
            #print dup.columns
            time = dup.iloc[0]['time']
            x = np.sum(dup['x'] * dup['area']) / total_area
            y = np.sum(dup['y'] * dup['area']) / total_area
            row = {'frame':f,
                   'time':time,
                   'area':total_area,
                   'x':x,
                   'y':y,
                   'contour_start_x':None,
                   'contour_start_y':None,
                   'contour_encode_len':None,
                   'contour_encoded':None}
            new_rows.append(row)
        new_rows = pd.DataFrame(new_rows)
        #print new_rows.head()
        relevant_data = pd.concat([relevant_data, new_rows])
        relevant_data.drop_duplicates('frame', take_last=True, inplace=True)

    relevant_data.set_index('frame', inplace=True)
    return relevant_data

def node_speed(node, experiment, graph):
    node_data = graph.consolidate_node_data(experiment=experiment, node=node)
    nd = curate_data(node_data)
    t, x, y = nd['time'], nd['x'], nd['y']
    t_spaced, speed = clean_speed(np.array(t), np.array(x), np.array(y))
    return t_spaced, speed

class SpeedWriter(object):

    def __init__(self, eid, window_size=11, gap_max_seconds=10, write_dir=None):
        self.eid = eid

        blob_output_dir = paths.output(eid)
        #print blob_output_dir
        self.blob_dir = blob_output_dir
        self.window_size = window_size
        self.gap_max_seconds = gap_max_seconds

        self._experiment = Experiment(fullpath=blob_output_dir,
                                      experiment_id=eid)
        #print 'experiment loaded from:', self._experiment.directory
        if write_dir is None:
            self.directory = paths.speed(eid)
        else:
            self.directory = write_dir

    def _pull_blob_positions(self, blob):
        if blob is None:
            return None
        full_df = blob.df
        if full_df is None:
            return None

        df = full_df[['frame', 'time']]
        x, y = zip(*full_df['centroid'])
        df['x'] = x
        df['y'] = y
        return df

    def speed_for_bid(self, bid, window_size=None):
        blob = self._experiment[bid]
        blob_df = self._pull_blob_positions(blob)
        split_df_list = self.split_dfs(blob_df)
        return self.combine_split_dfs(split_df_list, window_size)

    def iter_through_blob_dfs(self, min_points=60, max_count=None):
        e = self._experiment
        for i, (bid, blob) in enumerate(e.blobs()):
            df = self._pull_blob_positions(blob)

            if df is None: # do not yield empty blobs
                continue
            if len(df) < min_points: # do not yield short tracks
                continue
            yield (bid, df)

            # if max_count specified, stop yielding after max count
            if max_count is not None and i + 1 >= max_count:
                break

    def combine_split_dfs(self, df_list, window_size=None):
        if window_size is None:
            window_size = self.window_size
        blob_parts = []
        for b_df in df_list:
            if len(b_df) < window_size:
                continue
            df = self._speed_for_blob_df(b_df, window_size=window_size)
            blob_parts.append(df)
        return pd.concat(blob_parts)

    def split_dfs(self, blob_df, gap_max_seconds=None):

        if gap_max_seconds is None:
            gap_max_seconds = self.gap_max_seconds
        df_list = []
        t = np.array(blob_df['time'])
        dt = np.diff(t)
        gap_indicies = np.where(dt >= gap_max_seconds)[0]

        #print 'indicies', gap_indicies
        if len(gap_indicies) == 0:
            return [blob_df]

        last_gap_mid = -1
        for gap_index in gap_indicies:
            gap_len = t[gap_index + 1] - t[gap_index]
            assert gap_len >= gap_max_seconds
            gap_mid = (t[gap_index + 1] + t[gap_index]) / 2.0
            df = blob_df[(blob_df['time'] < gap_mid) & (blob_df['time'] >= last_gap_mid)]
            df_list.append(df)
            last_gap_mid = gap_mid

        df = blob_df[blob_df['time'] >= last_gap_mid]
        df_list.append(df)
        return df_list

    def _speed_for_blob_df(self, blob_df, window_size=None,
                           pad_ends=True, prelim_interp=True,
                           remove_outliers=True):

        if window_size is None:
            window_size = self.window_size
        t = np.array(blob_df['time'])
        y = np.array(blob_df['y'])
        x = np.array(blob_df['x'])
        #print blob_df
        half_win = window_size // 2

        prelim_interp = True
        if prelim_interp:
            interp_x = interpolate.interp1d(t, x, kind='linear')
            interp_y = interpolate.interp1d(t, y, kind='linear')

            time_segments = []

            dt = np.diff(t)
            break_points = np.where(dt > 1)[0]

            last_bp = 0
            for bp in break_points:
                t0, t1 = t[bp], t[bp+1]
                t_fill = np.linspace(start=t0, stop=t1,
                                     num=10 * int(dt[bp]))[1:-1]
                time_segments.append(t[last_bp:bp])
                time_segments.append(t_fill)
                last_bp = bp

            time_segments.append(t[last_bp:])

            t = np.concatenate(time_segments)
            x = interp_x(t)
            y = interp_y(t)
        if pad_ends:
            x0_pad = np.array([x[0] for i in range(half_win)])
            xN_pad = np.array([x[-1] for i in range(half_win)])

            y0_pad = np.array([y[0] for i in range(half_win)])
            yN_pad = np.array([y[-1] for i in range(half_win)])

            x_padded = np.concatenate((x0_pad, x, xN_pad))
            y_padded = np.concatenate((y0_pad, y, yN_pad))
            ts = t
            xs = smooth('boxcar', x_padded, window_size)
            ys = smooth('boxcar', y_padded, window_size)
        else:
            ts = t[half_win:-half_win]
            xs = smooth('boxcar', x, window_size)
            ys = smooth('boxcar', y, window_size)

        ti, xi = interpolate_to_1s(ts, xs)
        ti, yi = interpolate_to_1s(ts, ys)
        speed = calculate_speed(ti, xi, yi)

        df = pd.DataFrame()
        df['time'] = ti[1:]
        df['x'] = xi[1:]
        df['y'] = yi[1:]
        df['speed'] = speed
        if remove_outliers:
            df = self.iterativly_remove_outliers(df)
        return df

    def write_speed(self, bid, speed_df):
        if not self.directory.is_dir():
            self.directory.mkdir()
        file_name = '{bid}.csv'.format(bid=bid)
        file_path = self.directory / file_name
        speed_df.to_csv(str(file_path), index=False)

    def outlier_detection_df(self, df, col='speed', window_size = 21):
        good_rows = []
        outlier_rows = []

        half_win = 11 // 2
        N = len(df)

        if len(df) < window_size * 1.1:
            good_rows = list(df.index)
            outlier_rows = []
            return good_rows, outlier_rows

        # do not cut off front
        for i in range(half_win):
            row = df.iloc[i]
            row_name = row.name
            good_rows.append(row_name)


        for i in range(half_win, N - half_win):
            # row
            row = df.iloc[i]
            row_name = row.name
            x = row[col]

            # m is the number of standard deviations for 99% confidence (normal)
            m = 2.3263
            data = df[i - half_win: i + half_win][col]
            mean = np.mean(data)
            std = np.std(data)
            deviation = np.abs(x - mean) / std

            if deviation > m:
                outlier_rows.append(row_name)
            else:
                good_rows.append(row_name)

        # do not cut off back
        for i in range(N - half_win, N):
            row = df.iloc[-i]
            row_name = row.name
            good_rows.append(row_name)

        good_rows = sorted(list(set(good_rows)))
        outlier_rows = sorted(list(set(outlier_rows)))
        return good_rows, outlier_rows

    def iterativly_remove_outliers(self, df, col='speed'):
        for i in range(10):
            g, o = self.outlier_detection_df(df, col=col)
            if not len(o):
                break
            df = df.loc[g]
        return df

    def write_all_speeds(self, min_points):
        typical_bl = self._experiment.typical_bodylength
        for bid, blob_df in self.iter_through_blob_dfs(min_points=min_points):
            #speed_df = self._speed_for_blob_df(blob_df)
            split_df_list = self.split_dfs(blob_df)
            speed_df = self.combine_split_dfs(split_df_list)
            if not len(speed_df):
                print('WARNING', self.eid, bid, 'has no speeds')
                continue
            max_speed = max(speed_df['speed'])
            if max_speed > typical_bl:
                print('WARNING', self.eid, bid, 'showing unusually fast speeds')
                print(float(max_speed)/typical_bl, 'bl per s')
                #print('skipping')
                continue
            self.write_speed(bid, speed_df)

def pull_track_dfs(eid, start_time=0, min_time=10):
    """ returns a dictionary containing dataframes used for speed
    and position calculations.

    params:
    -----
    eid: (str)
        the experiment-id
    start_time: (int)
        the time in minutes from the begining of the recording
        that should be cut off of the front of all tracks.
    min_time: (int)
        the minimum amount of minutes a track should exist for in
        order to be included in the dictionary.
    """
    # data_id could be eventually included as a paramiter.
    # but more work would be needed to make all the scaling
    data_id = 'speed_files'
    track_dfs = {}
    wr = WormWriter(eid)
    sw = SpeedWriter(eid)

    typical_bl = sw._experiment.typical_bodylength
    for bid in wr.blobs(data_id=data_id):
        unique_id = '{eid}-{bid}'.format(eid=eid, bid=bid)
        sdf = wr.load(bid, data_id)
        sdf['minutes'] = sdf['time'] / 60.0
        df = sdf[sdf['minutes'] > start_time]
        if df is None or len(df) < 100:
            continue
        if  max(df['minutes']) - min(df['minutes']) > min_time:
            df = sw.iterativly_remove_outliers(df, col='speed')
            df['bl / s'] = df['speed'] / typical_bl
            track_dfs[unique_id] = df
    return track_dfs


def bootstrap_ks(df, col='bl / s', sample_frac=0.1, sample_num=100,
                 sample_size=None):
    """
    """
    scales = []
    locs = []
    if sample_frac is not None and sample_size is None:
        size = int(round(sample_frac * len(df)))
    elif sample_frac is None and sample_size is not None:
        size =  sample_size
    else:
        print('Warning: both sample size and sample fraction'
              'specified, using sample fraction')
    # fit exponential dist to many subsamples of data
    for i in range(sample_num):
        sample_df = df.loc[random.sample(df.index, size)]
        s = sample_df[col]
        loc, scale = stats.expon.fit(s, floc=0)
        locs.append(loc)
        scales.append(scale)
    # take mean of all fits
    mean_scale = np.mean(scales)
    mean_loc = np.mean(locs)

    # subsample data one more time. and test against mean fit.
    sample_df = df.loc[random.sample(df.index, size)]
    s = sample_df[col]
    ks_stat, p_val = stats.kstest(s, 'expon', args=(loc, scale))
    sample_size = size
    return p_val, sample_size, mean_loc, mean_scale


def get_ks_df_for_eid(eid, start_time = 30, sample_frac = 0.05, min_time=10):

    track_dfs = pull_track_dfs(eid=eid, start_time=start_time, min_time=min_time)

    test_data = []
    for track, df in track_dfs.items():

        if df is None or len(df) < 100:
            print(track, 'too few points')
            continue

        if  max(df['minutes']) - min(df['minutes']) < min_time:
            print(track, 'too few minutes')
            continue

        p_val, n, loc, scale = bootstrap_ks(df, sample_frac=sample_frac)
        data = {'eid':eid,
                'track': track,
                'p-val': p_val,
                'n-points': n,
                'loc':loc,
                'scale':scale,
                't0': min(df['minutes']),
                'tN': max(df['minutes']),
                'minutes': max(df['minutes']) - min(df['minutes'])}
        #print(data)
        test_data.append(data)
    ks_df = pd.DataFrame(test_data)
    return ks_df

def get_ks_df_for_eid_window_size(eid, start_time=30,
                                  sample_frac=0.05, window_size=10,
                                  sample_size=None):


    track_dfs = pull_track_dfs(eid=eid, start_time=start_time,
                               min_time=window_size)
    test_data = []
    for track, df in track_dfs.items():

        # keep the part of the track that is after start_time
        if df is None: continue
        df = df[df['minutes'] >= start_time]
        if len(df) < 100: continue

        end = max(df['minutes'])
        start = min(df['minutes'])

        if end -  start < window_size: continue

        # loop through all possible windows in track
        for i in range(1000):
            t0 = start + window_size * i
            t1 = start + window_size * (i + 1)

            if t1 > end: break

            locs = (df['minutes'] >= t0) & (df['minutes'] <= t1)
            window_df = df.loc[locs]

            p_val, n, loc, scale = bootstrap_ks(window_df,
                                                sample_frac=sample_frac,
                                                sample_size=sample_size)
            data = {'eid':eid,
                    'window':window_size,
                    'part': i,
                    'track': track,
                    'p-val': p_val,
                    'n-points': n,
                    'loc':loc,
                    'scale':scale,
                    'mean': np.nanmean(window_df['bl / s']),
                    't0': min(window_df['minutes']),
                    'tN': max(window_df['minutes']),
                    'minutes': (max(window_df['minutes']) -
                                min(window_df['minutes']))}

            test_data.append(data)
    ks_df = pd.DataFrame(test_data)
    return ks_df
