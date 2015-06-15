import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph


class BaseDataFrame(object):
    def __init__(self):
        self.df = None
        self.protected_cols = []

    def fill_gaps(self, df=None):
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

    def smooth_df(self, df=None, window=11, cols_to_smooth=None,
                  fill_gaps=True):

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
        def add_empty_rows(df, keys=None):
            cols = df.columns
            index = keys
            new_rows = np.empty((len(keys), len(cols)))
            new_rows.fill(np.NAN)
            df2 = pd.DataFrame(new_rows, index=index, columns=cols)
            return pd.concat([df, df2]).sort()

        if cols_to_smooth is None:
            cols_to_leave = set(self.protected_cols)
            # cols_to_leave = set(['minutes', 'time', 'frame',
            #                      'angle', 'orientation'])
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
        # if 'frame' in df:
        #     df.loc[:, 'frame'] = df['frame'].interpolate('nearest')
        # if 'orientation' in df:
        #   df.loc[:, 'orientation'] = df['orientation'].interpolate('nearest')
        # if 'angle' in df:
        #     df.loc[:, 'angle'] = df['angle'].interpolate('nearest')
        # if 'minutes' in df:
        #     df.loc[:, 'minutes'] = df.index / 60.0

        # Downsample to exact second times
        df = df.loc[t]
        df = df.reset_index()
        df = df.drop_duplicates('time')
        min_dif = np.nanmin(np.nanmin(np.diff(df['time'])))
        max_dif = np.nanmax(np.nanmax(np.diff(df['time'])))
        assert min_dif - max_dif < 0.000001  # assert almost equal
        return df

    def split_df(self, df=None, max_gap_seconds=1, max_gap_frames=None):

        if df is None:
            df = self.df

        if df.index.name is not None:
            df = df.reset_index()

        df_list = []
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


class Behavior_Coding(BaseDataFrame):
    def __init__(self, bl=None, body_length=None):
        self.raw_df = None
        self.df = None
        self.bl = bl
        self.moving_window_size = 11
        self.body_length = body_length
        self.protected_cols = ['time', 'frame']

        self.behavior_codes = {-1: 'unclassified',
                               0: 'pause',
                               1: 'forward',
                               2: 'back',
                               3: 'coil',
                               4: 'pirouette'}

        # not sure if this is wise?
        self.movement_codes = {-1: 'unclassified',
                               0: 'back',
                               1: 'pause',
                               2: 'forward'}

    def read_from_blob_df(self, blob_df):
        df = blob_df[['frame', 'time']].copy()

        x, y = zip(*blob_df.loc[:, 'centroid'])
        df['x'] = x
        df['y'] = y

        length, width = zip(*blob_df['size'])
        df.loc[:, 'length'] = length
        df.loc[:, 'width'] = width

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
            if len(rdf) > 11:
                df = self.fill_gaps(df=rdf)
                df = self.smooth_df(df=df)
                df = self.calculate_columns_df(df=df)
                df = self.smooth_df(df=df)
                df = self.equally_space_df(df=df, dt=dt)
                df_list.append(df)
        df = self.combine_split_dfs(df_list, min_window_seconds)
        self.df = df
        self.df.loc[:, 'behavior_class'] = -1
        self.df.loc[:, 'move_dir'] = -1

    def calculate_columns_df(self, df=None):

        if df is None:
            df = self.df

        if df.index.name is not None:
            df = df.reset_index()

        # calculate aditional columns
        df.loc[:, 'dx'] = df['x'].diff()
        df.loc[:, 'dy'] = df['y'].diff()

        df.loc[:, 'move_or'] = np.arctan2(df['dy'], df['dx'])
        df.loc[:, 'orientation'] = np.arctan2(df['len_y'], df['len_x'])
        self.fix_orientation(df)

        dt = df['time'].diff()
        move_dist = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)

        theta = (df['orientation'] - df['move_or'])

        df.loc[:, 'speed'] = move_dist / dt
        df.loc[:, 'speed_perp'] = move_dist * np.sin(theta) / dt
        df.loc[:, 'speed_along'] = move_dist * np.cos(theta) / dt

        df.loc[:, 'std_length'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)
        df.loc[:, 'd_angle'] = df['orientation'].diff()
        df.loc[:, 'angular_v'] = df['d_angle'] / df['time'].diff()
        df.loc[:, 'ar'] = df['width'] / df['length']
        df.loc[:, 'std_ar'] = df['std_width'] / df['std_length']
        df.loc[:, 'minutes'] = df['time'] / 60.0
        df.loc[:, 'minutes'] = df['time'] / 60.0
        # df = self.bound_angular_velocity(df=df)
        return df

    # def bound_angular_velocity(self, df=None):
    #     if df is None:
    #         df = self.df

    #     for _ in range(10):
    #         i = df[df['angular_v'] > 2*np.pi].index
    #         df.loc[i, 'angular_v'] = df.loc[i, 'angular_v'] - 2*np.pi
    #         i = df[df['angular_v'] < 2*np.pi].index
    #         df.loc[i, 'angular_v'] = df.loc[i, 'angular_v'] + 2*np.pi

    #     i = df[df['angular_v'] > np.pi].index
    #     df.loc[i, 'angular_v'] = df.loc[i]['angular_v'] - 2*np.pi

    #     i = df[df['angular_v'] < -np.pi].index
    #     df.loc[i, 'angular_v'] = df['angular_v'].loc[i] + 2*np.pi
    #     return df

    def fix_orientation(self, df, orientation_col='orientation'):
        df.loc[:, 'dorr'] = df[orientation_col].diff()
        df.loc[:, 'dorr'].fillna(0, inplace=True)
        df['dorr'].iloc[0] = df.iloc[0][orientation_col]

        pi = np.pi
        for i in range(10):
            big_minus = df['dorr'] < - pi
            df.loc[big_minus, 'dorr'] = df.loc[big_minus, 'dorr'] + (2 * pi)
            big_plus = df['dorr'] > pi
            df.loc[big_plus, 'dorr'] = df.loc[big_plus, 'dorr'] - (2 * pi)
        df.loc[:, orientation_col] = np.cumsum(df['dorr'])
        return df

    def _show_assignment(self, df, plot_col='ar',
                         assignment_col='behavior_class',
                         assignment_value=1, ax=None):
        # counts = [1,5,10]
        if ax is None:
            fig, ax = plt.subplots(figsize=(13, 3))
        # df.plot(x='time', y='speed', color='k', alpha=0.3, ax=ax)
        assigned = (df[assignment_col] == assignment_value)
        print(np.sum(assigned), 'assigned')
        not_assingned = not assigned  # (assigned == False)
        print(np.sum(not_assingned), 'not assigned')
        # print(not_paused)

        d1 = df[assigned]
        if d1 is not None and len(d1):
            ax.plot(np.array(d1['time']), np.array(d1[plot_col]), '.',
                    color='red', alpha=0.8, label='assinged')
        d2 = df[not_assingned]
        if d2 is not None and len(d2):
            ax.plot(np.array(d2['time']), np.array(d2[plot_col]), '.',
                    color='blue', alpha=0.8, label='not_assigned')
        ax.legend(loc=(1.1, 0.1))
        return ax

    def _assign_behavior(self, df, col_name='ar', value_cuttoff=0.8,
                         operation='>', min_points=5,
                         assignment_col='behavior_class',
                         assignment_value=1):
        parts = []
        if 'behavior_class' not in df.columns:
            df.loc[:, 'behavior_class'] = -1
            print('reset behavior')
        for df_part in self.split_df(df=df):
            if operation == '>':
                x = np.array(df_part[col_name] > value_cuttoff, dtype=int)
            elif operation == '>=':
                x = np.array(df_part[col_name] >= value_cuttoff, dtype=int)
            elif operation == '<':
                x = np.array(df_part[col_name] < value_cuttoff, dtype=int)
            elif operation == '<=':
                x = np.array(df_part[col_name] <= value_cuttoff, dtype=int)

            y = self.count_true_values(x)
            df_part.loc[y >= min_points, assignment_col] = assignment_value

            parts.append(df_part)
        df = self.combine_split_dfs(parts)
        return df

    def assign_coils(self, df, ar_cut=0.8, min_points=5):
        return self._assign_behavior(df,
                                     col_name='ar',
                                     value_cuttoff=ar_cut,
                                     operation='>',
                                     min_points=min_points,
                                     assignment_col='behavior_class',
                                     assignment_value=1)

    def assign_pauses(self, df, speed_cut=0.8, min_points=5):
        return self._assign_behavior(df,

                                     # if
                                     col_name='speed',
                                     operation='<',
                                     value_cuttoff=speed_cut,

                                     # for more than min_points, points
                                     min_points=min_points,

                                     # than assign col to value
                                     assignment_col='behavior_class',
                                     assignment_value=0)

    def show_pauses(self, df, ax=None):
        # counts = [1,5,10]
        if ax is None:
            fig, ax = plt.subplots(figsize=(13, 3))
        # df.plot(x='time', y='speed', color='k', alpha=0.3, ax=ax)
        paused = (df['behavior_class'] == 0)
        not_paused = not paused  # (paused == False)
        # print(not_paused)

        d1 = df[paused]
        if d1 is not None and len(d1):
            ax.plot(np.array(d1['time']), np.array(d1['speed']), '.',
                    color='red', alpha=0.5, label='paused')
        d2 = df[not_paused]
        if d2 is not None and len(d2):
            ax.plot(np.array(d2['time']), np.array(d2['speed']), '.',
                    color='blue', alpha=0.5, label='moving')

        ax.legend(loc=(1.1, 0.1))
        return ax

    def reassign_front_back(self, df=None, speed_cuttoff=1.0,
                            ar_cut=0.8, min_points=5):
        was_self = False
        if df is None:
            was_self = True
            df = self.df

        # Find coils -- front/back can change without warning during coils
        if 1 not in set(df['behavior_class']):
            df = self.assign_coils(df, ar_cut=ar_cut, min_points=min_points)

        # remove all coiled segments from consideration
        df_unclassified = df[df['behavior_class'] == -1]
        # store classified values to be added later.
        df_classified = df[df['behavior_class'] != -1]

        df_list = self.split_df(df=df_unclassified)
        parts = [df_classified]
        for df_part in df_list:
            pos = df_part[df_part['speed_along'] > speed_cuttoff]
            neg = df_part[df_part['speed_along'] < - speed_cuttoff]

            if len(neg) > len(pos):
                # flip the orientation of that segment
                df_part.loc[:, 'orientation'] = df_part['orientation'] + np.pi

                # recalculate all values that depend on orientation
                move_dist = np.sqrt(df_part['dx'] ** 2 + df_part['dy'] ** 2)
                theta = (df_part['orientation'] - df_part['move_or'])
                dt = df_part['time'].diff()

                df_part.loc[:, 'speed_perp'] = move_dist * np.sin(theta) / dt
                df_part.loc[:, 'speed_along'] = move_dist * np.cos(theta) / dt
                df_part.loc[:, 'd_angle'] = df_part['orientation'].diff()
                df_part.loc[:, 'angular_v'] = df_part['d_angle'] / dt
                move_dir = np.sign(np.array(df_part['speed_along']))
                df_part.loc[:, 'move_dir'] = move_dir

            parts.append(df_part)
        df = self.combine_split_dfs(parts)
        if was_self:
            self.df = df
        return df


class Worm_Shape(object):
    def __init__(self, letter_cache=None):
        self.ARBIRARY_CONVERSION_FACTOR = 48
        self.letter_cache = {}
        self.df = None
        self.outlines = None
        self.buffer_size = None
        self.contours = None

    def char_to_point_shifts(self, ch):
        byte = ord(ch) - self.ARBIRARY_CONVERSION_FACTOR
        assert byte <= 63, 'error:(%s) is not in encoding range' % ch
        assert byte >= 0, 'error:(%s) is not in encoding range' % ch
        point_shifts = np.zeros(shape=(3, 2))
        for count in range(3):
            value = (byte >> 4) & 3
            if value == 0:
                point_shifts[count] = np.array([-1, 0])
            elif value == 1:
                point_shifts[count] = np.array([1, 0])
            elif value == 2:
                point_shifts[count] = np.array([0, -1])
            elif value == 3:
                point_shifts[count] = np.array([0, 1])
            byte <<= 2
        return point_shifts

    def convert_outline_to_points(self, outline, length):
        length = int(length)
        point_arrays = [np.zeros(shape=[1, 2])]
        for i, ch in enumerate(outline):
            if ch in self.letter_cache:
                point_arrays.append(self.letter_cache[ch])
            else:
                char_point_shifts = self.char_to_point_shifts(ch)
                self.letter_cache[ch] = char_point_shifts
                point_arrays.append(char_point_shifts)
        point_shifts = np.concatenate(point_arrays, axis=0)
        point_shifts = point_shifts[:length + 1]
        points = np.cumsum(point_shifts, axis=0)

        # this code is purely to make sure outline forms a closed shape
        missing_distance = np.array(points[0] - points[-1])
        xy_steps = np.abs(missing_distance)
        total_steps = np.sum(xy_steps)
        if total_steps > 1:

            direction = (missing_distance / missing_distance *
                         np.sign(missing_distance))
            correction = np.ones(shape=(total_steps, 2)) * direction
            for i in range(int(total_steps)):
                if xy_steps[0] >= xy_steps[1]:
                    shift = np.array([1, 0])
                else:
                    shift = np.array([0, 1])
                correction[i] = correction[i] * shift
                xy_steps -= shift
                # print('xy steps')
                # print(xy_steps)

            last_point = points[-1].reshape(1, 2)
            new_point_shifts = np.concatenate([last_point, correction], axis=0)
            new_points = np.cumsum(new_point_shifts, axis=0)[1:]
            points = np.concatenate([points, new_points], axis=0)
            if np.sum(np.abs(np.array(points[0] - points[-1]))) > 1:
                print('total steps', total_steps)
                print('missing distance', missing_distance)
                print('start', points[0])
                print('end', points[-1])
                print('new_point_shifts')
                print(new_point_shifts)
                print('new_points')
                print(new_points)
                print('points')
                print(points[-6:])
                plt.plot(points[:, 0], points[:, 1], '.-')
                plt.show()
            start_end_dist = np.sum(np.abs(np.array(points[0] - points[-1])))
            assert start_end_dist <= 1, 'outline not closed loop'

        return points

    def read_blob_df(self, blob_df):
        d = blob_df.dropna(subset=['contour_encoded'])

        n_rows = len(d)
        x_midlines = np.zeros(shape=(n_rows, 11))
        y_midlines = np.zeros(shape=(n_rows, 11))
        for i, (_, row) in enumerate(d.iterrows()):
            xi, yi = zip(*row['midline'])
            x_midlines[i] = xi
            y_midlines[i] = yi

        outline_df = d[['frame', 'time']].copy()
        x, y = zip(*d['centroid'])
        outline_df.loc[:, 'centroid_x'] = np.round(x)
        outline_df.loc[:, 'centroid_y'] = np.round(y)
        x, y = zip(*d['contour_start'])
        outline_df.loc[:, 'contour_x'] = x
        outline_df.loc[:, 'contour_y'] = y

        x, y = zip(*d['std_vector'])
        x = np.array(x)
        y = np.array(y)
        # outline_df.loc[:, 'elipse_major_x'] = x
        # outline_df.loc[:, 'elipse_major_y'] = y

        length, width = zip(*d['size'])
        length, width = np.array(length), np.array(width)
        outline_df.loc[:, 'elipse_angle'] = np.arctan2(y, x)
        outline_df.loc[:, 'elipse_major'] = length
        outline_df.loc[:, 'elipse_minor'] = width

        list(blob_df.columns)
        outlines = []
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        for i, row in d[['contour_encode_len', 'contour_encoded']].iterrows():
            l = row['contour_encode_len']
            e = row['contour_encoded']

            if e is None:
                print('what?! inconceivable')
                break
            points = self.convert_outline_to_points(e, l)
            x, y = points[:, 0], points[:, 1]
            xmins.append(min(x))
            xmaxs.append(max(x))
            ymins.append(min(y))
            ymaxs.append(max(y))
            outlines.append(points)

        center_shift_x = (- outline_df.loc[:, 'contour_x']
                          + outline_df.loc[:, 'centroid_x'])
        center_shift_y = (- outline_df.loc[:, 'contour_y']
                          + outline_df.loc[:, 'centroid_y'])
        outline_df.loc[:, 'center_shift_x'] = center_shift_x
        outline_df.loc[:, 'center_shift_y'] = center_shift_y
        outline_df.loc[:, 'x_max'] = xmaxs - center_shift_x
        outline_df.loc[:, 'x_min'] = xmins - center_shift_x
        outline_df.loc[:, 'y_max'] = ymaxs - center_shift_y
        outline_df.loc[:, 'y_min'] = ymins - center_shift_y

        buffer_size = np.max(np.max(np.abs(outline_df[['x_max', 'x_min',
                                                       'y_max', 'y_min']])))
        self.df = outline_df
        self.buffer_size = buffer_size
        self.outlines = outlines
        self.x_mid = x_midlines
        self.y_mid = y_midlines

    def create_contours(self):

        buffer_size = self.buffer_size
        # np.max(np.max(np.abs(outline_df[['x_max', 'x_min',
        # 'y_max', 'y_min']])))
        outlines = self.outlines
        contours = np.zeros(shape=(len(outlines),
                                   2 * buffer_size + 1,
                                   2 * buffer_size + 1))

        lenght, xlim, ylim = contours.shape
        print(xlim, ylim)
        o = np.array(self.df[['center_shift_x', 'center_shift_y']])
        for i, outline in enumerate(outlines):
            shift = o[i]
            outline2 = outline - shift + np.array([buffer_size, buffer_size])
            for point in outline2:
                pt = point
                # TESTING, the contours are given in transposed coordinates...?
                # contours[i][pt[0], pt[1]] = 1
                contours[i][pt[1], pt[0]] = 1
        self.contours = contours

    def match_contour_frames(self, desired_frames):

        wdf = self.df.copy()
        wdf = wdf.reset_index()
        wdf = wdf.set_index('frame')
        wdf = wdf.loc[desired_frames]
        wdf[wdf['index'] >= len(self.contours)]
        wdf[wdf['index'] < 0]

        desired_contours = np.array(wdf.dropna('index')['index'])
        print(np.min(wdf['index']), 'min index')
        print(np.min(desired_contours))
        shape = list(self.contours.shape)
        shape[0] = len(desired_contours)

        # xshape = list(self.x_mid.shape)
        # xshape[0] = len(desired_contours)
        # x_mid2 = np.zeros(xshape)

        # yshape = list(self.y_mid.shape)
        # yshape[0] = len(desired_contours)
        # y_mid2 = np.zeros(yshape)

        contours2 = np.zeros(shape)
        for i, j in enumerate(desired_contours):
            if j > len(self.contours):
                print(j, 'wtf')
            if j < 0:
                print(j, 'wtf')
            contours2[i] = morph.binary_fill_holes(self.contours[j])
            # x_mid2[i] = self.x_mid[j]
            # y_mid2[i] = self.y_mid[j]
        return contours2

        # TODO: Add next steps into process.
        # contours2 = [morph.binary_fill_holes(c) for c in contours]

    def loop_thinning(self, contours2):
        thin = np.zeros(shape=contours2.shape)
        for i, c in enumerate(contours2):
            m = c.copy()
            for j in range(10):
                m, p1 = iterate_z(m, 0)
                m, p2 = iterate_z(m, 1)

                if not p1 and not p2:
                    break
            # thin.append(m)
            thin[i] = m
        return thin


def iterate_z(Z, subiteration=0):
    p = [Z[0:-2, 1:-1], Z[0:-2, 2:], Z[1:-1, 2:], Z[2:, 2:],
         Z[2:, 1:-1], Z[2:, 0:-2], Z[1:-1, 0:-2], Z[0:-2, 0:-2]]
    N = np.zeros(Z.shape, int)
    N[1:-1, 1:-1] = sum(p)
    check1 = (2 <= N) & (N <= 6)

    # count 01 edges, E
    E = np.zeros(Z.shape, int)
    p.append(p[0])

    pold = p[0]
    for pi in p[1:]:
        E[1:-1, 1:-1] += (pold == 0) & (pi == 1)
        pold = pi

    # if edge count ==1 consider for removal
    check2 = (E == 1)

    # HELTENA removed, we can use the defined above, offset -2!
    # p = [0, 0, Z[0:-2,1:-1], Z[0:-2,2:], Z[1:-1,2:], Z[2:  ,2:],
    #     Z[2:  ,1:-1], Z[2:  ,0:-2], Z[1:-1,0:-2], Z[0:-2,0:-2]]

    if subiteration == 0:
        p24 = p[2] * p[4]
        east_wind = np.zeros(Z.shape, int)
        # east_wind[1:-1,1:-1] = Z[ :-2,1:-1] * Z[1:-1,2:] * Z[2:  ,1:-1]
        # east_wind[1:-1,1:-1] = p[2] *p[4] * p[6]
        east_wind[1:-1, 1:-1] = p[0] * p24  # p[2] * p[4] # offset p[-2]!!
        check3 = (east_wind == 0)

        south_wind = np.zeros(Z.shape, int)
        # south_wind[1:-1,1:-1] = Z[1:-1,2:] * Z[2:  ,1:-1] * Z[1:-1, :-2]
        # south_wind[1:-1,1:-1] = p[4] * p[6] * p[8]
        south_wind[1:-1, 1:-1] = p24 * p[6]  # p[2] * p[4] * p[6] # offset p[-2]!!
        check4 = (south_wind == 0)
    else:
        p06 = p[0] * p[6]
        west_wind = np.zeros(Z.shape, int)
        # west_wind[1:-1,1:-1] = Z[ :-2,1:-1] * Z[1:-1,2:] * Z[1:-1, :-2]
        # west_wind[1:-1,1:-1] = p[2] *p[4] * p[8]
        west_wind[1:-1, 1:-1] = p06 * p[2]  # p[0] *p[2] * p[6] # offset p[-2]!!
        check3 = (west_wind == 0)

        north_wind = np.zeros(Z.shape, int)
        # north_wind[1:-1,1:-1] = Z[ :-2,1:-1] * Z[2:  ,1:-1] * Z[1:-1, :-2]
        # north_wind[1:-1,1:-1] = p[2] * p[6] * p[8]
        north_wind[1:-1, 1:-1] = p06 * p[4]  # p[0] * p[4] * p[6] # offset p[-2]!!
        check4 = (north_wind == 0)

    removal = check1 & check2 & check3 & check4
    Z1 = (Z == 1)
    removed = Z1 & (removal == 1)
    points_removed = removed.any()
    Z = np.array(Z1 & (removal == 0), int)
    return Z, points_removed
