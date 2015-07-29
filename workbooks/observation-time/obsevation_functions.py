import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiworm.core import MWTSummaryError
from waldo.output.speed import pull_track_dfs
from waldo.wio.experiment import Experiment
from waldo.behavior import Behavior_Coding
from waldo.wio import paths


def make_cdf(a, maxbins=10000):
    """ takes a list or array and returns the x,y values for plotting a cdf """
    x = np.sort(a)
    y = np.linspace(0, 1, len(a))
    return x, y

def pull_tracks_for_eids(eids, path, min_time=20, min_timepoints=5000, dt=1.0):
    """
    pulls behavior df data for tracks in a give experiment that match
    criterion

    path -- WALDO directory
    min_time -- minimum duration of track (minutes)
    min_timepoints -- minimum number of timepoints
    dt -- the equally spaced timestep (in seconds)
        that behavior df will be downsampled to match.
    """
    tracks = {}
    for eid in eids:
        cache = BehaviorCacher(eid=eid)
        data_type = 'behavior'
        cached_bids = cache.type_bids(data_type)
        if cached_bids:
           print('retreiving cached tracks for {eid}'.format(eid=eid))
           for bid in cached_bids:
               tracks[bid] = cache.read_df(bid=bid, data_type=data_type)

        if not cached_bids:
            try:
                #eid_tracks = pull_track_dfs(eid, start_time=start_time)
                pl = path / eid / 'blob_files'
                eid_tracks = pull_tracks_from_eid(eid, path=pl, min_time=20, min_timepoints=5000, dt=1.0)
                for bid in eid_tracks:
                    df = eid_tracks[bid]
                    print('writing', bid)
                    cache.write_df(bid=bid, data_type='behavior', df=df)

                # tracks = tracks + eid_tracks
                tracks.update(eid_tracks)
                print(eid, '---', len(eid_tracks), 'tracks')

            except MWTSummaryError:
                print(eid, '- skipping')
            except Exception as e:
                print(eid, 'skipped', e)

    print('\n', len(tracks), ' tracks in total')
    return tracks


def pull_tracks_from_eid(eid, path, min_time=20, min_timepoints=5000, dt=1.0):
    """
    eid - id
    pl - pathlib object to waldo directory
    min_time - (int) number of minutes track needs to be longer than in order to be considered.
    """
    #print(eid)
    #path = pl / eid / 'blob_files'
    #print(path)
    e = Experiment(fullpath=path, experiment_id=eid)
    print(e.id)
    typical_bodylength = e.typical_bodylength

    print(typical_bodylength, 'bl')
    b_dict = {}
    # b_list = []
    failed_dict = {}
    for i, (bid, blob) in enumerate(e.blobs()):
        blob_df = blob.df
        # print(blob_df.head())
        # print(i, len)
        # needs at least 5000 frames
        if len(blob_df) < min_timepoints:
            continue

        # needs to exist for at least min_time minutes
        t = blob_df['time']
        if t.iloc[-1] - t.iloc[0] < min_time * 60:
            continue

        # process blob
        try:
            bc = Behavior_Coding(bl=typical_bodylength, bid=bid)
            bc.read_from_blob_df(blob_df)
            bc.preprocess(dt=dt)
            df = bc.df
            df.loc[:, 'bl / s'] = df['speed'] / bc.bl
            # b_list.append(df)
            new_bid = '{eid}-{bid}'.format(eid=eid, bid=bid)
            b_dict[new_bid] = df
        except:
            failed_dict[bid] = len(blob_df)

    print(len(b_dict), 'blobs match criterion')
    print(len(failed_dict), 'blobs fail criterion')
    return b_dict

def calculate_error_window(df, key_col='avg'):
    """
    this function adds three aditional columns to a dataframe
    'error' -- difference between current value and final value
    '% error' -- % version of error
    'error_window' -- monatonically decreasing version of % error

    params
    -----
    df: (pd.DataFrame)
        blob timeseries df (must contain the key_col as one of columns)
    key_col: (str)
        the column in the dataframe used to calculate
    """
    # best_guess = df[key_col].iloc[-1]
    # df['error'] = np.abs(df[key_col] - best_guess)
    # df['% error'] = df['error'] / best_guess
    # df['error_window'] = df['% error'].copy()

    # for i in range(len(df)):
    #     df['error_window'].iloc[:-1] = np.nanmax(np.array([df['error_window'].iloc[:-1], df['error_window'].iloc[1:]]), axis=0)
    # return df

    d = df.copy()
    key_col = 'avg'
    best_guess = d[key_col].iloc[-1]
    d['error'] = np.abs(d[key_col] - best_guess)
    d['% error'] = d['error'] / best_guess
    #d['error_window'] = d['% error'].copy()
    a = np.array(d['% error'])
    for i in range(1000):
        a0 = a.copy()
        for j in range(10):
            a[:-1] = np.max([a[:-1], a[1:]], axis=0)
        if (a0 - a == 0).all():
            break

    d['error_window'] = a
    return d


def compile_error_curves(dfs, window_size = 60):
    """
    takes a list of timeseries dfs and
    returns a DataFrame in which each column is
    the monatonically decreasing version of % error
    for one of the dfs in the list.

    usefull for summarizing how a bunch of timeseries converge on
    some value after a certain point.

    params
    -----
    dfs: (list of pd.DataFrames)
        each df should be a track timeseries
    window_size: (int or float)
        size of bins (in seconds)
    """

    error_series = []
    for i, t in enumerate(dfs):
        df = dfs[t]
        df_window = df[df['t'] <= window_size].copy()
        if df_window is None:
            continue
        if len(df_window) < 0.8 * window_size:
            continue
        end_time = df_window.iloc[len(df_window)-1]['t']
        #print(t, len(df_window) / 60., end_time)

        d = calculate_error_window(df_window).set_index('t')['error_window']
        d = d.reindex(np.arange(0, window_size + 1))
        d = d.fillna(method='bfill')
        d = d.fillna(method='ffill')
        d.name = t
        error_series.append(d)
    return pd.concat(error_series, axis=1)


def add_crosspoint_line(cross_level, ax, d, color):
    """
    plot modification function



    """
    cross_point = np.where(d['error_window'] <= cross_level/100.)[0][0]
    max_minute = np.nanmax(np.array(d['minutes']))
    cross_min = d.iloc[cross_point]['minutes']
    ax.plot([cross_min, max_minute], [cross_level, cross_level], '--', alpha=0.8, color=color)
    ax.plot([cross_min, cross_min], [0, cross_level], '--', alpha=0.8, color=color)
    ax.plot([cross_min], [cross_level], 'o', color=color)


def plot_compiled(df):
    """
    makes a standard plot for a compiled_df that was created using the
    compile_error_curves function. As seen in Fig 3C

    params
    -----
    df: (pd.DataFrame)
        each column is the monatonically decreasing version of % error
        for one of the dfs in the list. the index is time in minutes.
    """
    fig = plt.figure(figsize=(8, 3))
    ax = plt.subplot(axisbg='white')
    minutes = np.array(df.index) / 60.0
    mean = df.mean(axis=1) * 100 # in this line df used to be compiled df.
    for i, col in enumerate(df):
        if i == 0:
            ax.plot(minutes, df[col] * 100, 'k', alpha=0.2, label='$R^M_i$ -- individual worms')
        ax.plot(minutes, df[col]* 100, 'k', alpha=0.2)

    ax.plot(minutes, mean, 'black', lw=4, label='mean $R^M_i$ -- all individuals' )

    cross_point20 = np.where(mean <= 20)[0][0] / 60.0
    ax.plot([cross_point20, cross_point20], [0, mean[int(cross_point20 * 60.0)]], '--', color='purple', lw=2)
    ax.plot([0, cross_point20], [mean[int(cross_point20 * 60.0)], mean[int(cross_point20 * 60.0)]],
            '--', color='purple', lw=2)
    ax.plot([cross_point20], [mean[int(cross_point20 * 60.0)]], 'o', color='purple', markersize=10)


    cross_point10 = np.where(mean <= 10)[0][0] / 60.0
    ax.plot([cross_point10, cross_point10], [0, mean[int(cross_point10 * 60.0)]], '--', color='darkred', lw=2)
    ax.plot([0, cross_point10], [mean[int(cross_point10 * 60.0)], mean[int(cross_point10 * 60.0)]],
            '--', color='darkred', lw=2)
    ax.plot([cross_point10], [mean[int(cross_point10 * 60.0)]], 'o', color='purple', markersize=10)


    cross_point5 = np.where(mean <= 5)[0][0] / 60.0
    ax.plot([cross_point5, cross_point5], [0, mean[int(cross_point5 * 60.0)]], '--', color='steelblue', lw=2)
    ax.plot([0, cross_point5], [mean[int(cross_point5 * 60.0)], mean[int(cross_point5 * 60.0)]],
            '--', color='steelblue', lw=2)
    ax.plot([cross_point5], [mean[int(cross_point5 * 60.0)]], 'o', color='steelblue', markersize=10)
    ax.legend(loc='best', frameon=False, fontsize=15)

    ax.set_ylabel('$R^M_i$ -- % relative error', size=15)
    ax.set_xlabel('$i$ -- time (minutes)', size=15)
    ax.set_ylim([0, 50])
    #plt.show()


def calculate_bar_df(compiled_df, percents = [5, 10, 20] ):
    """
    creates boxplot version of compiled df curves for multiple
    % errors. As seen in Fig 3D.

    params
    -----
    compiled_df: (pd.DataFrame)
        df output from the compile_error_curves function
        each column is the monatonically decreasing version of % error
        for one of the dfs in the list. the index is time in minutes.
    percents: (list of ints)
        the % error thresholds for which you want to check how long it takes for each
        track to reach.
    """
    rows = []
    for col in compiled_df:
        row = {'name':col}
        for p in percents:
            w = np.where(compiled_df[col] < p/100.)[0][0] / 60.0
            row[p] = w
        rows.append(row)

    bar_graph_df = pd.DataFrame(rows).T
    bar_graph_df.columns = bar_graph_df.loc['name']
    bar_graph_df = bar_graph_df.loc[percents]
    return bar_graph_df


class BehaviorCacher(object):
    def __init__(self, eid, write_dir=None):
        self.eid = eid

        blob_output_dir = paths.output(eid)
        #print blob_output_dir
        self.blob_dir = blob_output_dir
        self._experiment = Experiment(fullpath=blob_output_dir,
                                      experiment_id=eid)
        #print 'experiment loaded from:', self._experiment.directory
        if write_dir is None:
            self.directory = paths.behavior(eid)
        else:
            self.directory = write_dir

    def _file_path(self, bid, data_type):
        file_name = '{dt}_{bid}.csv'.format(dt=data_type, bid=bid)
        return self.directory / file_name
    def _type_glob(self, data_type):
        return list(self.directory.glob('{dt}*.csv'.format(dt=data_type)))

    def write_df(self, bid, data_type, df, **kwargs):
        if not self.directory.is_dir():
            self.directory.mkdir()
        file_path = self._file_path(bid, data_type)
        df.to_csv(str(file_path), index=False, **kwargs)

    def exists(self, bid, data_type):
        file_path = self._file_path(bid, data_type)
        return file_path.exists()

    def read_df(self, bid, data_type, **kwargs):
        file_path = self._file_path(bid, data_type)
        return pd.read_csv(str(file_path), **kwargs)

    def delete(self, bid, data_type):
        if self.exists(bid, data_type):
            file_path = self._file_path(bid, data_type)
            os.remove(str(file_path))

    def type_bids(self, data_type):
        return [str(f).split('{dt}_'.format(dt=data_type))[-1].split('.csv')[0]
                for f in self._type_glob('behavior')]