#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
#from scipy import stats
import scipy.interpolate as interpolate
import scipy.signal as ss
from waldo.wio import paths
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

    def __init__(self, ex_id, window_size=11, write_dir=None):
        self.ex_id = ex_id

        blob_output_dir = paths.output(ex_id)
        #print blob_output_dir
        self.blob_dir = blob_output_dir
        self.window_size = window_size

        self._experiment = Experiment(fullpath=blob_output_dir)
        #print 'experiment loaded from:', self._experiment.directory
        if write_dir is None:
            self.directory = paths.speed(ex_id)
        else:
            self.directory = write_dir

    def ensure_dir_exists(self):
        pass
        # make sure that self.directory exists and is ready to write to

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
        return self.speed_for_blob_df(blob_df, window_size=window_size)

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

    def speed_for_blob_df(self, blob_df, window_size=None):
        if window_size is None:
            window_size = self.window_size
        t = np.array(blob_df['time'])
        y = np.array(blob_df['y'])
        x = np.array(blob_df['x'])
        #print blob_df
        half_win = window_size // 2
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
        return df

    def write_speed(self, bid, speed_df):
        if not self.directory.is_dir():
            self.directory.mkdir()
        file_name = '{bid}.csv'.format(bid=bid)
        file_path = self.directory / file_name
        speed_df.to_csv(str(file_path), index=False)


    def write_all_speeds(self, min_points):
        for bid, blob_df in self.iter_through_blob_dfs(min_points=min_points):
            speed_df = self.speed_for_blob_df(blob_df)
            self.write_speed(bid, speed_df)
