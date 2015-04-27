import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools

#from waldo.conf import settings
#from waldo.wio.experiment import Experiment
from waldo.output.speed import SpeedWriter
from waldo.wio.worm_writer import WormWriter


class BinSelfTest(object):

    def __init__(self, speed_df, bid=None, worm_writer=None,
                 bin_sweep_step=5):
        """
        """
        # inputs
        self.bid = bid # id of blob
        self.ww = worm_writer # WormWriter object for reading/writing this stuff
        self.df = speed_df # pandas DataFrame with 'speed' and 'time' columns (time should be in minutes)
        self.bin_sweep_step_size = bin_sweep_step

        # shorthand for some input features
        self.t = np.array(self.df['time'])
        self.s = np.array(self.df['speed'])
        self.first_t = min(self.t)
        self.last_t = max(self.t)

        # when bins are assigned
        self.bin_size = None
        self.start_pos = None
        self.bin_num = None
        self.bin_starts = None
        self.bin_ends = None
        self.cropped_df = self.df
        self.is_cropped = False

    def test_bin_v_bin(self, min_data_fraction=0.9):
        """ currently defunt. tests bins against other bins of same size
        rather than against the final distribution.
        """
        bin_tests = []
        bins = range(self.bin_num)
        #print bins
        for bin_a, bin_b in itertools.combinations(bins, 2):

            bin_name = '{size}-{a}-{b}'.format(size=self.bin_size, a=bin_a, b=bin_b)

            df_a = self.df[self.df['bin'] == bin_a]
            df_b = self.df[self.df['bin'] == bin_b]

            #print bin_name
            #print df_a.head()
            #print df_b.head()
            ta = np.array(df_a['time'])
            tb = np.array(df_b['time'])
            sa = np.array(df_a['speed'])
            sb = np.array(df_b['speed'])

            if df_a is None or df_b is None:
                print('skipping', bin_name, 'because it contains no data')
                continue
            min_len = min([len(ta), len(tb)])
            if min_data_fraction:
                if min_len < min_data_fraction * self.bin_size:
                    #print 'skipping', bin_name, 'because it contains only', float(len(bin_df))/self.bin_size, '% bin data'
                    continue
            if min_len < 30:
                print('skipping', bin_name, 'because it contains only', min_len, 'data points')
                print(bin_a, len(ta))
                print(bin_b, len(tb))
                continue


            ks_stat, p_ks = stats.ks_2samp(sa, sb)
            row = {'bin-name': bin_name,
                   'dur': self.bin_size,
                   'bin-start-a': self.bin_starts[bin_a],
                   'bin-end-a': self.bin_ends[bin_a],
                   't-min-a': min(ta),
                   't-max-a': max(ta),
                   'N-a': len(ta),
                   'bin-start-b': self.bin_starts[bin_b],
                   'bin-end-b': self.bin_ends[bin_b],
                   't-min-b': min(tb),
                   't-max-b': max(tb),
                   'N-b': len(tb),
                   'ks': ks_stat,
                   'p-ks': p_ks,
                   }
            bin_tests.append(row)

        bin_comparison = pd.DataFrame(bin_tests)
        bin_comparison.set_index('bin-name', inplace=True)
        return bin_comparison

    def clear_bins(self):
        if 'bin' in self.df.columns:
            self.df.drop('bin', axis=1, inplace=True)
            self.cropped_df = self.df

        self.start_pos = None
        self.bin_size = None
        self.bin_num = None
        self.bin_starts = None
        self.bin_ends = None

    def characterize_bins(self, min_data_fraction=None):
        """
        """
        bin_tests = []

        if self.is_cropped:
            df = self.cropped_df
        else:
            df = self.df #[self.df['time'] >= self.start_pos]

        for b, bin_df in df.groupby('bin'):
            t = bin_df['time']
            s = bin_df['speed']
            bin_name = '{pos}_{size}-{b}'.format(pos=self.start_pos,
                                                 size=self.bin_size,
                                                 b=b)
            #print bin_name, len(bin_df)
            assert len(self.bin_starts) == self.bin_num
            #print self.bin_num
            #print self.bin_starts
            #print self.bin_ends

            #if self.bin_ends[b] > self.last_t:
            #    #$print max(t), self.last_t, self.bin_ends[b]
            #    continue

            if bin_df is None:
                print('skipping', bin_name, 'because it contains no data')
                continue
            if min_data_fraction:
                if len(bin_df) < min_data_fraction * self.bin_size:
                    #print 'skipping', bin_name, 'because it contains only', float(len(bin_df))/self.bin_size, '% bin data'
                    continue
            if len(bin_df) < 30:
                print('skipping', bin_name, 'because it contains only', len(bin_df), 'data points')
                continue

            row = {'bin-name': bin_name,
                   'dur': self.bin_size,
                   'start-pos':self.start_pos,
                   'bin-start': self.bin_starts[b],
                   'bin-end': self.bin_ends[b],
                   't-min': min(t),
                   't-max': max(t),
                   'is-cropped': self.is_cropped,
                   'N': len(t),
                   'mean': np.mean(s),
                   'median': np.median(s),
                   'std': np.std(s),
                   #'autocorr': np.correlate(s, s, mode='full'),
                   }
            bin_tests.append(row)

        bin_comparison = pd.DataFrame(bin_tests)
        if len(bin_comparison) < 1:
            return None
        bin_comparison.set_index('bin-name', inplace=True)
        return bin_comparison

    def test_bins(self, min_data_fraction=None):
        """
        """
        full = self.s
        bin_tests = []

        if self.is_cropped:
            df = self.cropped_df
        else:
            df = self.df #[self.df['time'] >= self.start_pos]

        for b, bin_df in df.groupby('bin'):
            t = bin_df['time']
            s = bin_df['speed']
            bin_name = '{pos}_{size}-{b}'.format(pos=self.start_pos,
                                                 size=self.bin_size,
                                                 b=b)
            #print bin_name, len(bin_df)
            assert len(self.bin_starts) == self.bin_num
            #print self.bin_num
            #print self.bin_starts
            #print self.bin_ends

            #if self.bin_ends[b] > self.last_t:
            #    #$print max(t), self.last_t, self.bin_ends[b]
            #    continue

            if bin_df is None:
                print('skipping', bin_name, 'because it contains no data')
                continue
            if min_data_fraction:
                if len(bin_df) < min_data_fraction * self.bin_size:
                    #print 'skipping', bin_name, 'because it contains only', float(len(bin_df))/self.bin_size, '% bin data'
                    continue
            if len(bin_df) < 30:
                print('skipping', bin_name, 'because it contains only', len(bin_df), 'data points')
                continue
            ks_stat, p_ks = stats.ks_2samp(full, s)

            # comment second line to stop anderson-darling and speed up performance A Lot.
            ad_stat, ad_crit_vals, p_ad = None, None, None
            #ad_stat, ad_crit_vals, p_ad = stats.anderson_ksamp([full, s])


            row = {'bin-name': bin_name,
                   'dur': self.bin_size,
                   'start-pos':self.start_pos,
                   'bin-start': self.bin_starts[b],
                   'bin-end': self.bin_ends[b],
                   't-min': min(t),
                   't-max': max(t),
                   'is-cropped': self.is_cropped,
                   'N': len(t),
                   'ks': ks_stat,
                   'p-ks': p_ks,
                   'ad': ad_stat,
                   'p-ad': p_ad,
                   }
            bin_tests.append(row)

        bin_comparison = pd.DataFrame(bin_tests)
        if len(bin_comparison) < 1:
            return None
        bin_comparison.set_index('bin-name', inplace=True)
        return bin_comparison

    def create_bin_size_sweep(self, start_t=0):
        df = self.df
        end_t = int(max(df['time']))
        #print 'end t', end_t
        #bin_sizes = range(5, end_t - start_t, 5)

        bin_sizes = []
        bin_step = self.bin_sweep_step_size
        for b in range(5, end_t - start_t + bin_step, bin_step):
            N = (end_t - start_t) / b
            if N == 1:
                bin_sizes.append(b)
                break
            bin_sizes.append(b)

        #print len(bin_sizes)
        start_positions = [start_t for i in bin_sizes]
        #print start_positions
        bin_array = zip(bin_sizes, start_positions)
        return bin_array

    def create_bin_and_start_sweep(self):
        #print self.first_t
        t0 = int(self.first_t)
        start_times = [i for i in [0, 10, 20, 30] if i > t0]
        start_times.insert(t0, 0)
        full_sweep = []
        for st in start_times:
            ba = self.create_bin_size_sweep(start_t=st)
            full_sweep.extend(ba)
        return full_sweep

    def characterize_bin_array(self, bin_array):
        """
        """
        bin_dfs = []
        for bin_size, start_pos in bin_array:
            #print 'bin size:', bin_size, '| start pos:', start_pos
            self.clear_bins()
            #print bin_size, start_pos, self.bin_num
            self.assign_bins(bin_size=bin_size, start_pos=start_pos)
            bin_df = self.characterize_bins()
            if bin_df is None:
                continue
            #print bin_df.head()
            bin_dfs.append(bin_df)
        return pd.concat(bin_dfs, axis=0)

    def test_bin_array(self, bin_array):
        """
        """
        bin_dfs = []
        for bin_size, start_pos in bin_array:
            #print 'bin size:', bin_size, '| start pos:', start_pos
            self.clear_bins()
            #print bin_size, start_pos, self.bin_num
            self.assign_bins(bin_size=bin_size, start_pos=start_pos)
            bin_df = self.test_bins()
            if bin_df is None:
                continue
            #print bin_df.head()
            bin_dfs.append(bin_df)
        return pd.concat(bin_dfs, axis=0)

    def write_bin_test(self, df):
        data_id = 'bin_size_self_test'
        bid = self.bid
        print(self.ww._filepath(bid, data_id))
        self.ww.dump(bid=bid, data_id=data_type, dataframe=df)


    def assign_bins(self, bin_size=5, start_pos=None):
        """
        """
        if start_pos is None:
            start_pos = int(self.first_t)

        # sort every timepoint into bins
        self.bin_size = bin_size
        self.start_pos = start_pos

        # is_cropped if the start position is after 0 and after first time-point
        is_cropped = (self.start_pos > 0) and (self.start_pos >= int(self.first_t))
        self.is_cropped = is_cropped
        # if cropping occurs, then make sure to use the cropped data for everything
        if is_cropped:
            self.cropped_df = self.df[self.df['time'] >= self.start_pos]
            times = self.cropped_df['time']
        else:
            self.cropped_df = self.df
            times = self.df['time']

        #print bin_size
        #print start_pos
        bin_assignment = [int((i-start_pos) // bin_size) for i in times]

        # theoretical starts and ends
        self.bin_num = max(bin_assignment) + 1
        self.bin_starts = [(i * bin_size) + start_pos for i in range(self.bin_num)]
        self.bin_ends = [(((i + 1) * bin_size) + start_pos - 1) for i in range(self.bin_num)]

        if min(bin_assignment) < 0:
            print('first last t', self.first_t, self.last_t)
            print('desired start pos', start_pos)
            print('num', self.bin_num)
            print('starts', self.bin_starts)
            print('ends', self.bin_ends)
            print('bins', set(bin_assignment))

        assert min(bin_assignment) >= 0
        assert len(self.bin_starts) == self.bin_num
        assert len(self.bin_ends) == self.bin_num

        if self.is_cropped:
            self.cropped_df['bin'] = bin_assignment
        else:
            self.df['bin'] = bin_assignment
        #print bin_size, start_pos, set(bin_assignment)

def self_bin_experiment(ex_id, minimum_minutes=40):
    sw = SpeedWriter(ex_id)
    wr = WormWriter(ex_id)
    acceptable_bids = []
    min_points = minimum_minutes * 60 * 10 / 4 # a low-ball estimate of how many points are in a track of X minutes
    for bid, df in sw.iter_through_blob_dfs(min_points=min_points, max_count=None):
        dt = max(df['time']) - min(df['time'])
        if dt > minimum_minutes * 60:
            acceptable_bids.append(bid)

    print(acceptable_bids)
    for bid in acceptable_bids:
        speed_df = sw.speed_for_bid(bid=bid)
        speed_df['time'] = speed_df['time'] / 60.0 # convert to min
        bst = BinSelfTest(speed_df, bid=bid, worm_writer=wr)
        sweep_bins = bst.create_bin_and_start_sweep()
        bin_results = bst.test_bin_array(sweep_bins)
        print(bin_results.head())
        bst.write_bin_test(bin_results)

    return acceptable_bids
