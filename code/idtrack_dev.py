import os
import sys
import pandas as pd
import numpy as np
import prettyplotlib as ppl
import matplotlib.pyplot as plt
from glob import glob
import h5py
import itertools
import setpath
os.environ.setdefault('WALDO_SETTINGS', 'default_settings')

from conf import settings
import images.worm_finder as wf
import wio.file_manager as fm
import prettyplotlib as ppl
import fingerprint.fingerprint as fp

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
        img_paths = glob(self.worm_dir + '/*_img.png')
        frame_path = [(int(ip.split('/')[-1].split('_')[0]), ip) for ip
                      in img_paths]

        frame_path = sorted(frame_path)
        self.im_frames = [f for (f,p) in frame_path]
        self.image_paths = [p for (f,p) in frame_path]

    def dump(self, frames, plus, minus):
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


# def compare_all_frames_same_worm(plus, minus):
#     l = len(plus)
#     index = range(l)
#     plus_dists = np.zeros((l,l))
#     minus_dists = np.zeros((l,l))
#     pairs = [p for p in itertools.combinations(index, 2)]
#     def fp_distance(array1, array2):
#         dif = np.fabs(array1-array2)
#         mean_dif = dif.mean()
#         return dif, mean_dif

#     for pair in pairs:
#         print pair
#         p0, p1 = pair
#         pdif, mean_pdif = fp_distance(plus[p0], plus[p1])
#         plus_dists[p0, p1] = mean_pdif
#         plus_dists[p1, p0] = mean_pdif

#         mdif, mean_mdif = fp_distance(minus[p0], minus[p1])
#         minus_dists[p0, p1] = mean_mdif
#         minus_dists[p1, p0] = mean_mdif

#         if False:
#             fig, ax = plt.subplots(2,3)
#             ax[0, 0].imshow(plus[p0])
#             ax[0, 1].imshow(plus[p1])
#             ax[0, 2].imshow(pdif)
#             ax[0,0].set_ylabel('i1 + i2')
#             ax[0,2].set_title('difference')

#             ax[1, 0].imshow(minus[p0])
#             ax[1, 1].imshow(minus[p1])
#             ax[1, 2].imshow(mdif)
#             ax[1,0].set_ylabel('|i1 - i2|')
#             plt.show()

#     fig, ax = plt.subplots()
#     ppl.pcolormesh(fig, ax, plus_dists)
#     ax.set_title('i1 + i2')
#     ax.set_xlabel('frame')
#     ax.set_ylabel('frame')

#     fig, ax = plt.subplots()
#     ppl.pcolormesh(fig, ax, minus_dists)
#     ax.set_title('|i1 - i2|')
#     ax.set_xlabel('frame')
#     ax.set_ylabel('frame')

#     plt.show()
#def assign(fingerprint, reference_stacks):

def best_fit(fprint, stack):
    distances = np.fabs(stack - fprint).mean(axis=1).mean(axis=1)
    min_dist = np.min(distances)
    return min_dist

#plus_reference_stacks = []
#minus_reference_stacks = []

def get_data(base_dir):
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



def test_distances(test_stack, reference_stacks):
    min_dists = np.zeros((len(test_stack), len(reference_stacks)), dtype=float)
    for i, p in enumerate(test_stack):
        for j, stack in enumerate(reference_stacks):
            min_dists[i, j] = best_fit(p, stack)
    df = pd.DataFrame(min_dists, columns= wids)
    return df

############## Code starts here.
# load all data.

wids, reference_p, reference_m, test_p, test_m = get_data(base_dir)

for test_num, test_worm in enumerate(wids):
    print 'testing', test_worm

    savename_p = '{wid}_dists_plus.csv'.format(wid=test_worm)
    savename_m = '{wid}_dists_minus.csv'.format(wid=test_worm)
    print savename_m
    print savename_p

    df = test_distances(test_stack=test_p[test_num], reference_stacks=reference_p)
    df.to_csv(savename_p)

    order = np.array(df).argsort(axis=1)
    accuracy = float((order[: ,0] == test_num).sum()) / len(order)
    print test_worm, 'plus', round(accuracy, ndigits=2), '%'


    df = test_distances(test_stack=test_m[test_num], reference_stacks=reference_m)

    df.to_csv(savename_m)
    order = np.array(df).argsort(axis=1)
    accuracy = float((order[: ,0] == test_num).sum()) / len(order)

    print test_worm, 'minus', round(accuracy, ndigits=2), '%'

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
