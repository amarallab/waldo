import os
import sys
import pandas as pd
import numpy as np
import prettyplotlib as ppl
import matplotlib.pyplot as plt
from glob import glob
import h5py
import collections

# import itertools
# # import setpath
# os.environ.setdefault('WALDO_SETTINGS', 'default_settings')
#
# from conf import settings
# import images.worm_finder as wf
# import wio.file_manager as fm
# import prettyplotlib as ppl
# #import fingerprint.fingerprint as fp

CUTOUT_DIR = 'home/projects/worm_movement/Data/cutouts/'
base_dir = '/home/projects/worm_movement/Data/cutouts/20130702_135704'
CUTOUT_DIR = base_dir

# ex_id = '20130614_120518'
# ex_id = '20130318_131111'
# ex_id = '20130702_135704' # many pics
# ex_id = '20130614_120518'

class FingerprintStack(object):
    """

    """
    def __init__(self, worm_id, datadir=CUTOUT_DIR):
        self.wid = worm_id
        self.base_dir = datadir
        self.worm_dir = os.path.join(self.base_dir,
                                     'worm_{wid}'.format(wid=worm_id))
        # initialize data
        self.frames = []
        self.plus = []
        self.minus = []
        self.im_frames = []
        self.image_paths = []
        self.refresh_images()

    def refresh_images(self):
        """
        stores paths to all png files that belong to a specific worm
        """
        img_paths = glob(self.worm_dir + '/*_img.png')
        frame_path = [(int(ip.split('/')[-1].split('_')[0]), ip) for ip
                      in img_paths]

        frame_path = sorted(frame_path)
        self.im_frames = [f for (f,p) in frame_path]
        self.image_paths = [p for (f,p) in frame_path]

    def dump(self, frames, plus, minus):
        """
        saves an hdf5 file with stack and fingerprint data.
        """
        #filename = 'fingerprint_stack.h5'
        filename = '{wid}_fingerprint_stack.h5'.format(wid=self.wid)
        filepath = os.path.join(self.base_dir, filename)
        print filename
        with h5py.File(filepath, 'w') as f:
            f.create_dataset(name='frames',
                             shape=frames.shape,
                             dtype=frames.dtype,
                             data=frames,
                             chunks=True,
                             compression='lzf')
            f.create_dataset(name='plus',
                             shape=plus.shape,
                             dtype=plus.dtype,
                             data=plus,
                             chunks=True,
                             compression='lzf')
            f.create_dataset(name='minus',
                             shape=minus.shape,
                             dtype=minus.dtype,
                             data=minus,
                             chunks=True,
                             compression='lzf')

    def load(self):
        """
        returns the values for all stack and fingerprint data
        stored in the hd5f file for this particular worm.
        """
        filename = '{wid}_fingerprint_stack.h5'.format(wid=self.wid)
        filepath = os.path.join(self.base_dir, filename)
        with h5py.File(filepath, 'r') as f:
            if u'frames' in f.keys():
                frames = np.array(f['frames'])
            if u'plus' in f.keys():
                plus = np.array(f['plus'])
            if u'minus' in f.keys():
                minus = np.array(f['minus'])
        return frames, plus, minus

    def compute_fingerprints(self):
        """
        calculate the fingerprint data for all images with known
        paths.
        """
        plus_stack = []
        minus_stack = []
        frames = []
        shape = (1, 1)

        for frame, ip in zip(self.im_frames, self.image_paths):
            print frame, ip
            p, m = fp.calculate_fingerprint(ip)
            #self.dump(frame, p, m)
            frames.append(frame)
            plus_stack.append(p)
            minus_stack.append(m)
            shape = p.shape

        frames = np.array(frames, dtype=float)
        n = len(frames)
        plus_3d = np.zeros((n, shape[0], shape[1]), dtype=float)
        minus_3d = np.zeros((n, shape[0], shape[1]), dtype=float)
        for i, (p, m) in enumerate(zip(plus_stack, minus_stack)):

            plus_3d[i] = p
            minus_3d[i] = m

        self.dump(frames, plus_3d, minus_3d)
        return frames, plus_3d, minus_3d


def best_fit(fprint, stack, begin_frame, end_frame):
    """
    returns the minimum distance between one 2d fingerprint array
    and a stack of reference fingerprints (ie. one 3d np.ndarray

    params
    -----
    fprint: (np.ndarray)
        a 2d np array containing a fingerprint
    stack: (np.ndarray)
        a 3d np array containing many fingerprint stacks
    """
    values = np.fabs(stack[begin_frame:end_frame, :, :] - fprint)
    if len(values) == 0:
        return -1

    distances = values.mean(axis=1).mean(axis=1)
    min_dist = np.min(distances)
    return min_dist


def get_data(base_dir):
    """
    shorthand function to split all worms into test and reference
    stacks for plus and minus fingerprints.

    params
    -----
    base_dir: (str)
        the path to where all the fingerprint hdf5 files are saved.

    returns
    ------

    wids: (list)
         list of worm ids in the same order as they are arranged in all
        other outputs
    reference_p: (list of np.ndarrays)
        list of all reference np.ndarrays for plus pixel intensity
    reference_m: (list of np.ndarrays)
        list of all reference np.ndarrays for minus pixel intensity
    reference_p: (list of np.ndarrays)
        list of all test np.ndarrays for plus pixel intensity
    reference_m: (list of np.ndarrays)
        list of all test np.ndarrays for minus pixel intensity
    """

    print os.path.isdir(base_dir)
    #wormdirs = glob(base_dir + '/worm_*')
    fingerprint_stacks = glob(base_dir + '/*_finger*')
    #print fingerprint_stacks

    reference_p = []
    reference_m = []
    test_p = []
    test_m = []
    wids = []

    for fp in fingerprint_stacks[:]:
        path, filename = os.path.split(fp)
        worm_id = filename.split('_')[0]
        #print worm_id
        wids.append(worm_id)

        f = FingerprintStack(worm_id)
        frames, p, m = f.load()
        n = len(frames)
        sep = n / 2

        reference_p.append(p[:sep])
        test_p.append(p[sep:])

        reference_m.append(m[:sep])
        test_m.append(m[sep:])

    return wids, reference_p, reference_m, test_p, test_m



def test_distances(wids, test_stack, reference_stacks, begin_frame=None, end_frame=None):
    """
    returns dataframe with the minimum distances between each row
    of the test_stack and every stack in the reference stacks.

    params
    -----
    test_stack: (np.ndarray)
        a single np.ndarray containing all fingerprints that should
        be compared against each of the reference stacks.

    reference_stacks: (list of np.ndarrays)
          all reference np.ndarrays (image num, x, y) that should
          be compared against
    """
    eff_test_stack = test_stack[begin_frame:end_frame]
    min_dists = np.zeros((len(eff_test_stack), len(reference_stacks)), dtype=float)
    for i, p in enumerate(eff_test_stack):
        for j, stack in enumerate(reference_stacks):
            min_dists[i, j] = best_fit(p, stack, begin_frame, end_frame)
    df = pd.DataFrame(min_dists, columns=wids)
    return df



############## Code starts here. ##############

class FinishLoopException(Exception):
    pass


def main_window_size():
    WINDOW_SIZE_LIST = [1000, 200, 50]

    loop_count = 0
    max_loop_count = 20

    # load all data.
    wids, reference_p, reference_m, test_p, test_m = get_data(base_dir)

    results = []
    try:
        for window_size in WINDOW_SIZE_LIST:
            print reference_p[0].shape
            for test_num, test_worm in enumerate(wids):
                test_p_stack = test_p[test_num]
                test_m_stack = test_m[test_num]
                frame_count = len(test_p_stack)
                for begin_frame in range(0, frame_count, window_size):
                    end_frame = begin_frame + window_size

                    current_frame_count = len(test_p_stack[begin_frame:end_frame])
                    if current_frame_count == 0:
                        continue

                    print ':::: Window size: {size}[{begin}:{end}], testing {worm} ::::'.format(size=window_size,
                                                                                                begin=begin_frame,
                                                                                                end=end_frame,
                                                                                                worm=test_worm)

                    # savename_p = '{wid}_dists_plus.csv'.format(wid=test_worm)
                    # savename_m = '{wid}_dists_minus.csv'.format(wid=test_worm)
                    # print savename_m
                    # print savename_p
                    #
                    df = test_distances(wids=wids,
                                        test_stack=test_p_stack, reference_stacks=reference_p,
                                        begin_frame=begin_frame, end_frame=end_frame)

                    # df.to_csv(savename_p)

                    order = np.array(df).argsort(axis=1)
                    acc_p = float((order[:, 0] == test_num).sum()) / len(order)
                    print 'worm: {worm} PLUS, acc: {acc}%'.format(worm=test_worm, acc=round(acc_p * 100.0, ndigits=2))
                    # print df

                    df = test_distances(wids=wids,
                                        test_stack=test_m_stack, reference_stacks=reference_m,
                                        begin_frame=begin_frame, end_frame=end_frame)

                    # df.to_csv(savename_m)
                    order = np.array(df).argsort(axis=1)
                    acc_m = float((order[:, 0] == test_num).sum()) / len(order)
                    best, _ = collections.Counter(order[:, 0]).most_common()[0]

                    print "Order: ", order[:, 0]
                    print "Best: ", best

                    print 'worm: {worm} MINUS, acc: {acc}%'.format(worm=test_worm, acc=round(acc_m * 100.0, ndigits=2))
                    # print df

                    # row = {'window_size': window_size,
                    #        'begin_frame': begin_frame,
                    #        'end_frame': end_frame,
                    #        'wid': test_worm,
                    #        'acc_p': acc_p,
                    #        'acc_m': acc_m}
                    results.append([window_size, begin_frame, end_frame, current_frame_count, test_worm, acc_p, acc_m, wids[best]])

                    loop_count += 1
                    if loop_count >= max_loop_count:
                        raise FinishLoopException()
    except FinishLoopException, ex:
        pass


    results_df = pd.DataFrame(results, columns=('window_size', 'begin_frame', 'end_frame', 'frame_count', 'wid', 'acc_p', 'acc_m', 'best'))
    results_df.to_csv('acc_window_size_results.csv')

                # fig, ax = plt.subplots()
                # for l, w in zip(wids, worm_dists):
                #     if w != test_worm:
                #         ppl.plot(ax, w, label=l, alpha=0.5)
                #     else:
                #         ppl.plot(ax, w, label=l, alpha=0.5)

                # title = test_worm + ' ' + str(accuracy) + '% correct'
                # ax.set_title(title)
                # ax.legend()
                # #plt.show()
                # plt.savefig(test_worm + '_id_test.png')


def main_window_best_assignment():
    WINDOW_SIZE_LIST = [200, 50]

    loop_count = 0
    max_loop_count = 5

    # load all data.
    wids, reference_p, reference_m, test_p, test_m = get_data(base_dir)
    # if len(set([len(x) for x in test_p]) | set([len(x) for x in test_m])) != 1:
    #     print 'E: test_p and test_m have not the same item sizes'
    #     return
    frame_count = len(test_p[0])

    try:
        results = []
        for window_size in WINDOW_SIZE_LIST:
            print reference_p[0].shape
            for begin_frame in range(0, frame_count, window_size):
                end_frame = begin_frame + window_size
                assigned_wids = {}
                # assigned_wids[13286] = 12
                current_reference_p = reference_p[:]
                current_reference_m = reference_m[:]
                remain_wids = wids[:]
                while len(assigned_wids) < len(wids):
                    for test_num, test_worm in enumerate(wids):
                        test_p_stack = test_p[test_num]
                        test_m_stack = test_m[test_num]

                        current_frame_count = len(test_p_stack[begin_frame:end_frame])
                        if current_frame_count == 0:
                            assigned_wids[test_worm] = (None, None)
                            continue

                        df = test_distances(wids=remain_wids,
                                            test_stack=test_p_stack, reference_stacks=current_reference_p,
                                            begin_frame=begin_frame, end_frame=end_frame)

                        # df.to_csv(savename_p)

                        order = np.array(df).argsort(axis=1)
                        acc_p = float((order[:, 0] == test_num).sum()) / len(order)

                        df = test_distances(wids=remain_wids,
                                            test_stack=test_m_stack, reference_stacks=current_reference_m,
                                            begin_frame=begin_frame, end_frame=end_frame)

                        order = np.array(df).argsort(axis=1)
                        best, _ = collections.Counter(order[:, 0]).most_common()[0]
                        acc_m = float((order[:, 0] == test_num).sum()) / len(order)

                        assigned_wids[test_worm] = (remain_wids[best], acc_m)
                        results.append([window_size, begin_frame, end_frame, current_frame_count, test_worm, acc_p, acc_m, remain_wids[best]])
                        print 'Window size: {size}[{begin}:{end}], {worm:>5s} assigned {assigned:>5s} (acc: {acc})'\
                            .format(size=window_size,
                                    begin=begin_frame,
                                    end=end_frame,
                                    worm=test_worm,
                                    assigned=remain_wids[best],
                                    acc=acc_m)

                        current_reference_p.pop(best)
                        current_reference_m.pop(best)
                        remain_wids.pop(best)

                loop_count += 1
                if loop_count >= max_loop_count:
                    raise FinishLoopException()
    except FinishLoopException, ex:
        pass

    results_df = pd.DataFrame(results, columns=('window_size', 'begin_frame', 'end_frame', 'frame_count', 'wid', 'acc_p', 'acc_m', 'best'))
    results_df.to_csv('acc_window_best_assignment_results.csv')

                # fig, ax = plt.subplots()
                # for l, w in zip(wids, worm_dists):
                #     if w != test_worm:
                #         ppl.plot(ax, w, label=l, alpha=0.5)
                #     else:
                #         ppl.plot(ax, w, label=l, alpha=0.5)

                # title = test_worm + ' ' + str(accuracy) + '% correct'
                # ax.set_title(title)
                # ax.legend()
                # #plt.show()
                # plt.savefig(test_worm + '_id_test.png')



# main_window_size()  # First algorithm

main_window_best_assignment()
