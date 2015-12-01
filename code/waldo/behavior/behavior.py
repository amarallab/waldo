import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph
from scipy.stats import norm
def perp(v):
    # adapted from http://stackoverflow.com/a/3252222/194586
    p = np.empty_like(v)
    p[0] = -v[1]
    p[1] = v[0]
    return p

def circle_3pt(a, b, c):
    """
    1. Make some arbitrary vectors along the perpendicular bisectors between
        two pairs of points.
    2. Find where they intersect (the center).
    3. Find the distance between center and any one of the points (the
        radius).
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # find perpendicular bisectors
    ab = b - a
    c_ab = (a + b) / 2
    pb_ab = perp(ab)
    bc = c - b
    c_bc = (b + c) / 2
    pb_bc = perp(bc)

    ab2 = c_ab + pb_ab
    bc2 = c_bc + pb_bc

    # find where some example vectors intersect
    # center = seg_intersect(c_ab, c_ab + pb_ab, c_bc, c_bc + pb_bc)

    A1 = ab2[1] - c_ab[1]
    B1 = c_ab[0] - ab2[0]
    C1 = A1 * c_ab[0] + B1 * c_ab[1]
    A2 = bc2[1] - c_bc[1]
    B2 = c_bc[0] - bc2[0]
    C2 = A2 * c_bc[0] + B2 * c_bc[1]
    center = np.linalg.inv(np.matrix([[A1, B1], [A2, B2]])) * np.matrix([[C1], [C2]])
    center = np.array(center).flatten()
    radius = np.linalg.norm(a - center)
    return center, radius

class BaseDataFrame(object):
    def __init__(self):
        self.df = None
        self.protected_cols = []

    def fill_gaps(self, df=None):
        """
        uses linear interpolation to fill in any missing values in a dataframe
        """
        if df is None:
            df = self.df

        if df.index.name is not None:
            df = df.reset_index()

        # Fill small gaps using frames
        df = df.set_index(['frame'])
        f0, f1 = int(df.index[0]), int(df.index[-1])
        df = df.reindex(range(f0, f1))
        df = df.interpolate('linear')
        df = df.reset_index()
        return df

    def smooth_df(self, df=None, window=11, cols_to_smooth=None):
        """

        df - dataframe (default is primary df)
        window - (int) N points in running window
        cols_to_smooth - list of cols to smooth. default = all
        """
        # fill_gaps=True):

        if df is None:
            df = self.df

        if df.index.name is not None:
            df = df.reset_index()

        if cols_to_smooth is None:
            cols_to_leave = set(self.protected_cols)
            cols_to_smooth = list(set(df.columns) - cols_to_leave)

        # Fill small gaps using frames
        # if fill_gaps:
        #     df = df.set_index(['frame'])
        #     f0, f1 = int(df.index[0]), int(df.index[-1])
        #     df = df.reindex(range(f0, f1))
        #     df = df.interpolate('linear')
        #     df = df.reset_index()

        df[cols_to_smooth] = pd.rolling_mean(df[cols_to_smooth],
                                             window=window, center=True)
        df = df.fillna(method='bfill')
        df = df.fillna(method='ffill')
        return df

    def equally_space_df(self, df=None, dt=0.5, cols_to_smooth=None,
                         key_col='time'):
        # """
        # df - dataframe. None = default df
        # cols_to_smooth - list of column names. None = all
        # key_col - column name to be used
        # dt - (float) timestep to use for equal spacing
        # """

        def add_empty_rows(df, keys=None):
            cols = df.columns
            index = keys
            new_rows = np.empty((len(keys), len(cols)))
            new_rows.fill(np.NAN)
            df2 = pd.DataFrame(new_rows, index=index, columns=cols)
            return pd.concat([df, df2]).sort()

        if cols_to_smooth is None:
            cols_to_leave = set(self.protected_cols)
            cols_to_smooth = list(set(df.columns) - cols_to_leave)

        if df.index.name is not None:
            df = df.reset_index()

        # Calculate sample times that fit timeseries
        scale_factor = 1.0 / dt
        t0 = np.ceil(df['time'].iloc[0] * scale_factor) / scale_factor
        t1 = np.floor(df['time'].iloc[-1] * scale_factor) / scale_factor
        # print(t0, t1)
        n = (t1 - t0) * scale_factor + 1
        t = np.round(np.linspace(t0, t1, n), decimals=1)

        # Add rows to fit exact sample times.
        df = df.set_index(['time'])
        df = add_empty_rows(df, keys=t)
        df.index.name = 'time'

        # Interpolate missing values
        df.loc[:, cols_to_smooth] = df[cols_to_smooth].interpolate('linear')
        for col in self.protected_cols:
            if col in df:
                df.loc[:, col] = df[col].interpolate('nearest')

        # Downsample to exact second times
        df = df.loc[t]
        df = df.reset_index()
        if 'time' not in df.columns:
            df['time'] = df['index']
        df = df.drop_duplicates('time')
        min_dif = np.nanmin(np.nanmin(np.diff(df['time'])))
        max_dif = np.nanmax(np.nanmax(np.diff(df['time'])))
        assert min_dif - max_dif < 0.000001  # assert almost equal
        return df

    def split_df(self, df=None, max_gap_seconds=1, max_gap_frames=None):

        """
        splits a dataframe into a list of smaller dataframes.
        use max_gap_seconds or max_gap_frames to determine gaps (not both)

        df - dataframe.
        max_gap_seconds -
        max_gap_frames -
        """

        if df is None:
            df = self.df

        if df.index.name is not None:
            df = df.reset_index()

        df_list = []
        # print(df.columns)
        # print(df.index.name)
        if df.index.name == 'time':
            t = np.array(df['time'])
        else:
            t = np.array(df['time'])
        dt = np.diff(t)
        gap_indicies = np.where(dt > max_gap_seconds)[0]

        if len(gap_indicies) == 0:
            return [df]

        last_gap_mid = -1
        for gap_index in gap_indicies:
            gap_len = t[gap_index + 1] - t[gap_index]
            assert gap_len >= max_gap_seconds
            gap_mid = (t[gap_index + 1] + t[gap_index]) / 2.0
            df_part = df[(df['time'] < gap_mid) & (df['time'] >= last_gap_mid)]
            df_list.append(df_part)
            last_gap_mid = gap_mid

        df_part = df[df['time'] >= last_gap_mid]
        df_list.append(df_part)
        return df_list

    def combine_split_dfs(self, df_list, min_window_seconds=0):
        """
        Combines a list of dfs into one df. (inverse of split_df)
        dfs should not have any overlapping timepoints.
        really short dfs (shorter than min_window_seconds) not included in final

        min_window_seconds - (float) minimum time a df must cover
        """

        # if list contains one element, return element
        if len(df_list) == 1:
            return df_list[0]

        blob_parts = []
        for df_part in df_list:
            if not len(df_part):
                continue
            t = df_part['time']
            t0, t1 = t.iloc[0], t.iloc[-1]
            # print('combine', t0, 'to', t1)
            seconds = t1 - t0
            if seconds < min_window_seconds:
                continue

            blob_parts.append(df_part)
        combined = pd.concat(blob_parts)
        combined = combined.sort('time')
        # print('min diff', np.nanmin(np.diff(combined['time'])))
        # print('max diff', np.nanmax(np.diff(combined['time'])))
        assert np.diff(combined['time']).all() > 0
        return combined

    def count_true_values(self, x):
        """
        takes an array of booleans and creates an array of ints.
        ints in array = then number of

        is this right? check function!!

        """
        for i in range(len(x)):
            x[(x[1:] > i) & (x[:-1] > i)] = i + 2
            if np.max(x) < i:
                break

        y = x.copy()
        for i in range(len(x)):
            y[1:] = np.max([y[1:], y[:-1]], axis=0)
            y[x == 0] = 0
            if np.max(y) < i:
                break
        return y

class Basic_Orientation_Prep(BaseDataFrame):
    def __init__(self, bid=None):
        self.bid = bid
        self.raw_df = None
        self.df = None

        self.moving_window_size = 11
        self.protected_cols = ['time', 'frame']

    def read_from_blob_df(self, blob_df):
        """
        takes blob_df from Nick's code and extracts key columns into
        flat df with no nested catagories.

        gets: frame, time, x, y, length, width, len_x, len_y, angle, std width
        """

        df = blob_df[['frame', 'time']].copy()

        # x, y = zip(*blob_df.loc[:, 'centroid'])
        # df['x'] = x
        # df['y'] = y

        length, width = zip(*blob_df['size'])
        df.loc[:, 'box_length'] = length
        df.loc[:, 'box_width'] = width

        len_x, len_y = zip(*blob_df['std_vector'])
        df.loc[:, 'len_x'] = len_x
        df.loc[:, 'len_y'] = len_y
        df.loc[:, 'angle'] = np.arctan2(len_y, len_x)

        df.loc[:, 'std_width'] = blob_df['std_ortho']

        self.raw_df = df
        self.df = df.copy()

    def preprocess(self, dt=0.2, max_gap_seconds=5, min_window_seconds=5):
        raw_df = self.raw_df
        raw_df_list = self.split_df(raw_df, max_gap_seconds)
        df_list = []
        for rdf in raw_df_list:
            if len(rdf) < 11:
                continue
            df = self.smooth_df(df=rdf, cols_to_smooth=['box_length', 'box_width'])
            df.loc[:, 'ar'] = df['box_width'] / df['box_length']

            # get the orientation calculations set up properly
            orientation_col = 'orr'
            df.loc[:, orientation_col] = df['angle']
            df.loc[:, 'dorr'] = df[orientation_col].diff()

            # fill missing values left by diff
            df.loc[:, 'dorr'].fillna(0, inplace=True)
            df['dorr'].iloc[0] = df.iloc[0][orientation_col]

            pi = np.pi
            # the  range(10)  is just a number of loops that will be larger than
            # the number of corrections required
            for i in range(10):
                big_minus = df['dorr'] < - pi
                df.loc[big_minus, 'dorr'] = df.loc[big_minus, 'dorr'] + (2 * pi)
                big_plus = df['dorr'] > pi
                df.loc[big_plus, 'dorr'] = df.loc[big_plus, 'dorr'] - (2 * pi)
            df.loc[:, orientation_col] = np.cumsum(df['dorr'])
            df['sm_dorr'] = df['dorr'].copy()
            df = self.smooth_df(df=df, cols_to_smooth=['sm_dorr'])
            df = self.equally_space_df(df=df, dt=dt)

            # calculate which parts might be coils
            coil_thresh = 0.4
            coil_min_dur = 10
            potential_coils = df['ar'] > coil_thresh
            pc = self.count_true_values(np.array(potential_coils, dtype=int))
            df['is_coil'] = (pc > coil_min_dur)

            # calculate which parts might be coils
            re_thresh = 0.01
            re_min_dur = 1
            potential_reorients = np.abs(df['sm_dorr']) > re_thresh
            pr = self.count_true_values(np.array(potential_reorients, dtype=int))
            df['is_reorienting'] = (pr > re_min_dur)

            df_list.append(df)
        df = self.combine_split_dfs(df_list, min_window_seconds)
        self.df = df

    # def calculate_columns_df(self, df=None):

    #     if df is None:
    #         df = self.df

    #     if df.index.name is not None:
    #         df = df.reset_index()

    #     # calculate aditional columns
    #     orientation_col = 'orientation'
    #     # df.loc[:, 'orientation'] = df['angle']
    #     df.loc[:,orientation_col] = df['angle']
    #     df.loc[:, 'dorr'] = df[orientation_col].diff()

    #     # fill missing values left by diff
    #     df.loc[:, 'dorr'].fillna(0, inplace=True)
    #     df['dorr'].iloc[0] = df.iloc[0][orientation_col]

    #     pi = np.pi
    #     for i in range(10):
    #         big_minus = df['dorr'] < - pi
    #         df.loc[big_minus, 'dorr'] = df.loc[big_minus, 'dorr'] + (2 * pi)
    #         big_plus = df['dorr'] > pi
    #         df.loc[big_plus, 'dorr'] = df.loc[big_plus, 'dorr'] - (2 * pi)
    #     df.loc[:, orientation_col] = np.cumsum(df['dorr'])

    #     df.loc[:, 'd_angle'] = df['orientation'].diff()
    #     df.loc[:, 'angular_v'] = df['d_angle'] / df['time'].diff()
    #     # df.loc[:, 'std_ar'] = df['std_width'] / df['std_length']
    #     df.loc[:, 'minutes'] = df['time'] / 60.0
    #     return df


class Basic_Shape_Prep(BaseDataFrame):
    def __init__(self, bid=None, coil_df=None):
        self.bid = bid
        self.raw_x_df = None
        self.raw_y_df = None
        self.x_df = None
        self.y_df = None
        self.df = None
        self.coil_df = coil_df

        self.moving_window_size = 11
        self.protected_cols = ['time', 'frame']

    def read_from_blob_df(self, blob_df):
        """
        takes blob_df from Nick's code and extracts key columns into
        flat df with no nested catagories.

        gets: frame, time, x, y, length, width, len_x, len_y, angle, std width
        """

        # grab basic df
        df = blob_df[['frame', 'time']].copy()
        length, width = zip(*blob_df['size'])
        df.loc[:, 'length'] = length
        df.loc[:, 'width'] = width
        self.basic_df = df

        # make seperate x, y dfs for midline
        d = blob_df.dropna(subset=['midline'])

        n_rows = len(d)
        x_rows = []
        y_rows = []

        for i, (_, row) in enumerate(d.iterrows()):
            xi, yi = zip(*row['midline'])
            rx = {i:j for i, j in enumerate(xi)}
            ry = {i:j for i, j in enumerate(yi)}
            rx['frame'] = ry['frame'] =row['frame']
            rx['time'] = ry['time'] = row['time']
            x_rows.append(rx)
            y_rows.append(ry)

        self.raw_x_df = pd.DataFrame(x_rows)
        self.raw_y_df = pd.DataFrame(y_rows)

        # save a midline df to modifiy
        self.x_df = self.raw_x_df.copy()
        self.y_df = self.raw_y_df.copy()

    def preprocess_midline_dfs(self, dt=0.2, max_gap_seconds=1, window_size=5):
        """
        splits on gaps, interpolates missing vals, smooths in time, equally spaces points in time
        """
        def preprocess_a_midline(mid_df, dt=dt, mgf=max_gap_seconds, ws=window_size):
            mid_df_list = self.split_df(mid_df, max_gap_seconds=mgf)
            df_list = []
            for mdf in mid_df_list:
                if len(mdf) > ws:
                    mdf = self.smooth_df(df=mdf, window=ws)
                    mdf = self.equally_space_df(df=mdf, dt=dt)
                    df_list.append(mdf)
            mdf = self.combine_split_dfs(df_list, min_window_seconds=5)
            mdf = self._allign_pos_matrix(mdf)
            return mdf

        self.x_df = preprocess_a_midline(self.x_df, mgf=max_gap_seconds, ws=window_size)
        self.y_df = preprocess_a_midline(self.y_df, mgf=max_gap_seconds, ws=window_size)

    def _allign_pos_matrix(self, x_df):
        # convert df to numpy array
        x_np = np.array(x_df[range(11)])
        x_flip_np = x_np[::, ::-1]

        # calculate diff, normal-v-normal and normal-v-flipped
        dx = np.abs(x_np[:-1] - x_np[1:]).sum(axis=1)
        dx_rev = np.abs(x_np[:-1] - x_flip_np[1:]).sum(axis=1)

        flip_switch = dx > dx_rev
        front_or_back = 1
        new_x = np.zeros(shape=x_np.shape, dtype=float)
        new_x[0] = x_np[0]
        for i, flip in enumerate(flip_switch):
            if flip:
                front_or_back = - front_or_back
            new_x[i+1] = x_np[i+1, ::front_or_back]

        # convert numpy array back to df
        x_df2 = x_df.copy()
        x_df2.loc[:,range(11)] = new_x
        return x_df2

    def length_calculations(self):
        """
        """
        # Everything for length calculations!
        x_np = self.x_df[range(11)]
        y_np = self.y_df[range(11)]
        len_df = self.y_df[['time', 'frame']]

        dx = np.diff(x_np, axis=1)
        dy = np.diff(y_np, axis=1)

        length = np.sqrt(dx ** 2 + dy ** 2).sum(axis=1)
        mean, var = norm.fit(length)
        is_len_good = np.abs(length - mean) < (2 * var)
        reliable_length = length[is_len_good]

        len_df.loc[:, 'length'] = length
        len_df.loc[:, 'length_is_good'] = is_len_good

        median1 = np.median(length)
        median2 = np.median(reliable_length)

        self.body_length = median2
        self.df = len_df

    def _calculate_curvature(self, worm_x, worm_y):
        xx = worm_x[0:2].tolist()
        yy = worm_y[0:2].tolist()
        radius_list = []
        for cx, cy in zip(worm_x[2:], worm_y[2:]):
            xx.append(cx)
            yy.append(cy)
            a, b, c = zip(xx, yy)
            try:
                center, radius = circle_3pt(a, b, c)
                inv_radius = 1.0 / radius
            except np.linalg.LinAlgError, er:
                inv_radius = 0.0
            radius_list.append(inv_radius)
            xx.pop(0)
            yy.pop(0)
        return np.array(radius_list)

    def _calculate_curvature_array(self, list_x, list_y):
        result = []
        for worm_x, worm_y in zip(list_x, list_y):
            radius_list = self._calculate_curvature(worm_x, worm_y)
            result.append(radius_list)
        return np.array(result)

    def compute_curvature(self):
        x = np.array(self.x_df[range(11)])
        y = np.array(self.y_df[range(11)])

        curveature = self._calculate_curvature_array(x, y)
        avg_curve = curveature.mean(axis=1)

        self.df.loc[:, 'avg_curve'] = avg_curve

    def find_head_tail(self, max_gap_seconds = 1, orr_df = None):

        x_df = self.x_df.copy()
        y_df = self.y_df.copy()
        if 'time' in x_df.columns:
            x_df.set_index('time', inplace=True)
        if 'time' in y_df.columns:
            y_df.set_index('time', inplace=True)
        if 'time' in orr_df.columns:
            orr_df.set_index('time', inplace=True)

        # make sure the indicies from different data frames line up.
        non_coil_times = set(orr_df[orr_df['is_coil'] == False].index)
        shape_times = set(x_df.index)
        good_times = non_coil_times.union(shape_times)
        sorted_good_times = sorted(list(good_times))

        # only select times that are not sorted.
        x_df = x_df.loc[sorted_good_times]
        y_df = y_df.loc[sorted_good_times]
        print(x_df.columns)
        print(x_df.index.name)
        x_df.index.name = 'time'
        y_df.index.name = 'time'
        y_df.reset_index(inplace=True)
        x_df.reset_index(inplace=True)
        print(x_df.columns)
        print(x_df.index.name)
        split_x = self.split_df(x_df, max_gap_seconds)
        split_y = self.split_df(y_df, max_gap_seconds)

        df = self.df.copy().set_index('time')
        dfs = []
        for xdf, ydf in zip(split_x, split_y):

            # convert to numpy
            x_np = np.array(xdf[range(10)])
            y_np = np.array(ydf[range(10)])

            # grab first and last columns
            front_x = x_np[:, 0]
            back_x = x_np[:, -1]
            front_y = y_np[:, 0]
            back_y = y_np[:, -1]

            front_dx = np.diff(front_x)
            back_dx = np.diff(back_x)
            front_dy = np.diff(front_y)
            back_dy = np.diff(back_y)

            # squared dist traveled for front and back.
            front_dist_2 = front_dx ** 2 + front_dy ** 2
            back_dist_2 = back_dx ** 2 + back_dy ** 2

            front_bigger = front_dist_2 >= back_dist_2
            head_x = front_x
            head_y = front_y

            if (float(sum(front_bigger)) / len(front_bigger)) < 0.5:
                head_x = back_x
                head_y = back_y

            times = xdf['time']
            frames = xdf['frame']
            df_part = df.loc[times].copy()
            df_part.loc[:, 'head_x'] = head_x
            df_part.loc[:, 'head_y'] = head_y
            df_part.index.name = 'time'
            dfs.append(df_part)

        df2 = pd.concat(dfs, axis=0)
        df2.loc[:, 'head_angle'] = np.arctan2(df2['head_y'], df2['head_x'])
        self.df2 = df2

class Basic_Speed_Prep(BaseDataFrame):
    def __init__(self, bid=None, bl=None, head_df=None, orr_df=None):
        self.bid = bid
        self.raw_df = None
        self.df = None
        self.bl = bl

        self.moving_window_size = 11
        self.protected_cols = ['time', 'frame']

    def read_from_blob_df(self, blob_df):
        """
        takes blob_df from Nick's code and extracts key columns into
        flat df with no nested catagories.

        gets: frame, time, x, y, length, width, len_x, len_y, angle, std width
        """

        df = blob_df[['frame', 'time']].copy()

        x, y = zip(*blob_df.loc[:, 'centroid'])
        df['x'] = x
        df['y'] = y

        self.raw_df = df
        self.df = df.copy()

    def preprocess(self, dt=0.2, max_gap_seconds=5, min_window_seconds=5):

        raw_df = self.raw_df
        raw_df_list = self.split_df(raw_df, max_gap_seconds)
        df_list = []
        for rdf in raw_df_list:
            if len(rdf) > 11:
                df = self.fill_gaps(df=rdf)
                df = self.smooth_df(df=df)
                df = self.calculate_columns_df(df=df)
                df = self.smooth_df(df=df)
                df = self.equally_space_df(df=df, dt=dt)
                df_list.append(df)
        df = self.combine_split_dfs(df_list, min_window_seconds)
        self.df = df

    def calculate_columns_df(self, df=None):

        if df is None:
            df = self.df

        if df.index.name is not None:
            df = df.reset_index()

        df.loc[:, 'dx'] = df['x'].diff()
        df.loc[:, 'dy'] = df['y'].diff()

        df.loc[:, 'move_or'] = np.arctan2(df['dy'], df['dx'])

        dt = df['time'].diff()
        dt = dt.fillna(method='bfill')
        dt = dt.fillna(method='ffill')
        move_dist = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)

        df.loc[:, 'speed'] = move_dist / dt / self.bl
        df.loc[:, 'minutes'] = df['time'] / 60.0
        return df


def analyze_a_blob(blob_df,bid, blobs_path):

    basic_orr = Basic_Orientation_Prep()
    basic_orr.read_from_blob_df(blob_df=blob_df)
    basic_orr.preprocess()

    print('basic orr done')
    basic_shape = Basic_Shape_Prep(coil_df=basic_orr.df[['is_coil', 'is_reorienting']])
    basic_shape.read_from_blob_df(blob_df=blob_df)
    basic_shape.preprocess_midline_dfs()
    basic_shape.length_calculations()
    basic_shape.compute_curvature()
    basic_shape.find_head_tail(orr_df=basic_orr.df)
    body_length = basic_shape.body_length
    print('basic shape done')

    basic_speed = Basic_Speed_Prep(bl=body_length)
    basic_speed.read_from_blob_df(blob_df=blob_df)
    basic_speed.preprocess()

    print('basic speed done')
    # check if orientation agrees/disagrees with head/tail
    # This should be silo'd off somewhere
    def normalize_an_orientation(df, col_name):
        for i in range(10):
            big_minus = df[col_name] < - np.pi
            df.loc[big_minus, col_name] = df.loc[big_minus, col_name] + (2 * np.pi)
            big_plus = df[col_name] > np.pi
            df.loc[big_plus, col_name] = df.loc[big_plus, col_name] - (2 * np.pi)

    df_s = basic_shape.df2.reset_index()[['time', 'head_x', 'head_y', 'head_angle',
                                        'length', 'length_is_good', 'avg_curve']]
    df_o = basic_orr.df.reset_index()[['time', 'frame', 'orr', 'is_coil', 'is_reorienting',
                                    'sm_dorr', 'box_length', 'box_width', 'ar']]

    df = pd.merge(df_s, df_o, on='time')

    df.loc[:, 'head_orr_diff'] = np.abs(df['head_angle'] - df['orr'])
    df.loc[:, 'head_matches_orr'] = True


    df.loc[:,'norm_orr'] = df['orr'].copy() + 4 * np.pi
    col_name = 'norm_orr'
    normalize_an_orientation(df, col_name)

    # flag segments were orientation is directed in opposite way that head is.
    is_wrong_way = (np.pi / 2 < df['head_orr_diff'] ) & (df['head_orr_diff'] < 3 * np.pi / 2)
    df.loc[is_wrong_way, 'head_matches_orr'] = False
    is_wrong_way = (5 * np.pi / 2 < df['head_orr_diff'] ) & (df['head_orr_diff'] < 7 * np.pi / 2)
    df.loc[is_wrong_way, 'head_matches_orr'] = False

    # make the new measurement
    df.loc[:, 'forward_angle'] = df['norm_orr']
    df.loc[df['head_matches_orr'] == False, 'forward_angle'] = - df['norm_orr']


    print('head correction done')

    df_speed = basic_speed.df.copy()[['time', 'x', 'y', 'speed', 'move_or']]
    df2 = pd.merge(df, df_speed, on='time')
    # print(len(df2))
    # df2 = df2.drop_duplicates('time')
    # print(len(df2))

    theta = (df2['forward_angle'] - df2['move_or'])

    df2.loc[:, 'speed_perp'] = df2['speed'] * np.sin(theta)
    df2.loc[:, 'speed_along'] = df2['speed'] * np.cos(theta)

    out_path = blobs_path.parents[0]   / 'clean_timeseries' #[0]
    if not out_path.exists():
        out_path.mkdir()
        print('made outpath: {p}='.format(p= out_path))
    outname = '{op}/bid{bid}-bl{bl}.csv'.format(op=out_path, bid=bid, bl=round(body_length,ndigits=2))
    df2.to_csv(outname)
    print('wrote: {o}'.format(o=outname))
