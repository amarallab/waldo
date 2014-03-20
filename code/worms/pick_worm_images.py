#!/usr/bin/env python

"""
Filename: pull_all_data_for_worm.py
Description:

The purpose of this code is to consolodate all required data to make
a discriptive figure focused on a single worm. This includes an image of the plate
and several timeseries.
"""

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import json
import glob
import random
import bisect
import math
import numpy as np
import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../shared/')
PROJECT_DIR = os.path.abspath(HERE + '/../../')
sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR)


print SHARED_DIR
#DATA_DIR = HERE + '/../Data/'

from wio.file_manager import WORM_DIR
from annotation.experiment_index import Experiment_Attribute_Index
from imagehandeling.grab_images import grab_images_in_time_range
from wormmetrics.measurement_switchboard import find_data, pull_blob_data
from Encoding.decode_outline import decode_outline

def pull_timeseries(blob_id, savedir):
    ''' 
    saves several timeseries files for blob_id:

    xy_raw position
    decoded_outlines
    centroid speeds by bl
    curvature by bl
    lengths
    '''

    metrics = ['xy_raw', 'curvature_all_bl', 'centroid_speed_bl', 'smooth_length', 'encoded_outline', 'smoothed_spine']
    #metrics = ['xy_raw']
    #metrics = ['encoded_outline']
    '''
    for metric in metrics:
        t, data = write_metric_json(blob_id, metric, savedir)
    start_time, end_time = t[0], t[-1]
    print start_time, end_time
    return start_time, end_time
    '''

def pull_plate_images(blob_id, savedir, start_time, end_time=3600.0, max_images=4):
    ''' 
    saves the first image of the plate for which the blob is present
    returns the time used in the image
    '''
    
    ex_id = '_'.join(blob_id.split('_')[:2])
    time_to_image = create_image_directory(ex_id)
    print '{N} images found for ex_id'.format(N=len(time_to_image))
    image_times = []
    for im_time, im_path in time_to_image.iteritems():        
        if start_time <= float(im_time) <= end_time:
            image_times.append(im_time)


    if len(image_times) < max_images:
        max_images = len(image_times)
    print '{N} potential images found'.format(N=len(image_times))
    selected_times = random.sample(image_times, max_images)
    print selected_times
    for im_time in selected_times:
        im_path = time_to_image[im_time]
        savename = '{sd}/{t}-plate.png'.format(t=im_time, sd=savedir.rstrip('/'))
        cmd = 'cp "{ip}" {sn}'.format(ip=im_path, sn=savename)
        print cmd
        os.system(cmd)
    return image_times

def ex_ids_matching_request(dataset, label, day, verbose=True):
    ei = Experiment_Attribute_Index()
    dataset_ids = ei.return_ex_ids_with_attribute('dataset', dataset)
    label_ids = ei.return_ex_ids_with_attribute('label', label)
    day_ids = ei.return_ex_ids_with_attribute('age', day)
    print 'dataset: {d} | label: {l} | day {y}'.format(d=len(dataset_ids), l=len(label_ids),
                                                       y=len(day_ids))
    merge = set(dataset_ids) & set(label_ids) & set(day_ids)
    print 'found {N} recordings matching request'.format(N=len(merge))

    if verbose:
        for ex_id in merge:
            atrib = ei.return_attributes_for_ex_id(ex_id)
            print '\t{id} | {d} | {l} | {y}'.format(id=ex_id, d=atrib['dataset'], 
                                                    l=atrib['label'], y=atrib['age'])
    return list(merge)

def find_all_worms(dataset, label, day, verbose=True, path=WORM_DIR):
    ex_ids = ex_ids_matching_request(dataset, label, day)
    all_blobs = []
    if verbose:
        print 'searching: {path}'.format(path=path)
    for eID in ex_ids:
        tag = '-spine'
        search_path = '{path}{eID}'.format(path=path, eID=eID)
        if not os.path.isdir(search_path):
            if verbose:
                print '{eID} not processed'.format(eID=eID)
            continue
        search = '{path}/*{t}*'.format(path=search_path.rstrip('/'), t=tag)
        #search = search_path + '/*'
        #print search
        blob_files = glob.glob(search)
        blobs = [b.split('/')[-1].split(tag)[0] for b in blob_files]
        if verbose:
            print '\t{eID} | {N} blobs'.format(eID=eID, N=len(blobs))
        all_blobs += blobs
    print '{N} blobs found matching criterion'.format(N=len(all_blobs))            
    return all_blobs

def pick_images_for_worm(blob_id, start_time=None, end_time=None):
    """    
    Arguments:
    - `blob_id`:
    """
    ex_id = '_'.join(blob_id.split('_')[:2])
    if not start_time or not end_time:
        times, spines = pull_blob_data(blob_id, metric='spine')
        start_time, end_time = times[0], times[-1]
    image_times, image_paths = grab_images_in_time_range(ex_id, start_time, end_time)
    return image_times, image_paths

def pick_N_drawable_worms(blob_ids, N=12):
    chosen_blobs = {}
    N_ids = len(blob_ids)
    # keep testing if randomly chosen image works until all blob ids are tried
    for i in xrange(N_ids):
        bID = blob_ids.pop(random.randint(0, N))
        N_ids = len(blob_ids)
        # grab the spine for this blob, if no spine, skip it.
        #times, spines = pull_blob_data(blob_id=bID, metric='spine')
        times, spines, _ = find_data(blob_id=bID, metric='spine')
        if len(times) == 0:
            print bID, 'has no spines'
            continue
        # grab the outline for this blob, if no spine, skip it.
        otimes, outlines, _ = find_data(blob_id=bID, metric='encoded_outline')
        #print otimes
        if len(otimes) == 0:
            print bID, 'has no outlines'
            continue
        # grab the spine for this blob, if no spine, skip it.
        start_time, end_time = times[0], times[-1]
        im_times, im_paths = pick_images_for_worm(blob_id=bID, start_time=start_time, end_time=end_time)
        if len(im_times) == 0:
            print bID, 'has no images'
            continue
        # only use image times that are not in a break (ie. have a nearby spine time)
        ok_times, paths = [], []
        for t, p in zip(im_times, im_paths):
            t = float(t)
            j = bisect.bisect_left(times, t)
            if math.fabs(times[j] -t) < 0.1:
                ok_times.append(t)
                paths.append(p)
        # only use image times that are not in a break (ie. have a nearby spine time)
        if len(ok_times) == 0:
            print bID, 'has no good image options'
            continue

        im_time, im_path = random.choice(zip(ok_times, paths))
        si = bisect.bisect_left(times, im_time)
        stime = times[si]
        spine = spines[si]
        oi = bisect.bisect_left(otimes, im_time)
        otime = otimes[oi]
        outline = outlines[oi]
        chosen_blobs[bID] = {'im_time':im_time,
                             'im_path':im_path,
                             'stime': stime,
                             'spine':spine,
                             'otime':otime,
                             'outline':outline}
        print '{bID} is good | {N} found'.format(bID=bID, N=len(chosen_blobs))
        if len(chosen_blobs) >= N:
            break
    return chosen_blobs

def save_drawable_worms(dataset, label, day, N=12, savename='hey.json'):
    worm_ids = find_all_worms(dataset, label, day) #, path=path)
    worm_summary = pick_N_drawable_worms(blob_ids=worm_ids,N=N)
    print len(worm_summary), 'worms found'
    for w in worm_summary:
        print w
    json.dump(worm_summary, open('pickworms_summary.json', 'w'))
    return worm_summary

def draw_worms(worms_to_draw, plot_name=None):
    fig = plt.figure()
    N_rows = 3
    N_cols = 4

    box_size = 100

    # create a grid of axis to draw on.
    gs1 = gridspec.GridSpec(N_rows, N_cols)
    gs1.update(left=0.01, right=0.99, wspace=0.00)
    ax1 = plt.subplot(gs1[0, 0])
    #ax = [plt.subplot(gs1[i%N_rows, i / N_rows], sharey=ax1, sharex=ax1) for i in range(1, 12)]
    ax = [plt.subplot(gs1[i%N_rows, i / N_rows]) for i in range(1, 12)]
    ax = [ax1] + ax

    for i, (blob_id, data) in enumerate(worms_to_draw.iteritems()):
        print i, blob_id
        im_path = data['im_path']
        img = Image.open(im_path)
        print 'imgsize', img.size
        background = np.array(img)#.T #.reshape(img.size[0], img.size[1])

        xy, l, code = data['outline']
        x, y = xy
        outline = decode_outline((x, y, l, code))
        x, y = zip(*outline)
        x_mid = (max(x) - min(x))/2 + min(x)
        y_mid = (max(y) - min(y))/2 + min(y)
        
        x_box = [x_mid - box_size/2, x_mid + box_size/2]
        y_box = [y_mid - box_size/2, y_mid + box_size/2]
        print x_box
        print y_box

        print 'size A:', background.shape
        background = background[x_box[0]:x_box[1], y_box[0]:y_box[1]]
        print 'size B:', background.shape
        #x = np.array(x) - x_box[0]
        #y = np.array(y) - y_box[0]
        #ax[i].plot(x, y)
        ax[i].imshow(background.T, cmap=cm.Greys_r)


    # remove x axis ticklabels from all but first box
    for a in ax:
        for tick in a.get_yticklabels():
            tick.set_fontsize(0.0)
        for tick in a.get_xticklabels():
            tick.set_fontsize(0.0)
    #plt.show()
    if plot_name:
        plt.savefig(plot_name)
    plt.clf()

if __name__ == '__main__':
    dataset = 'disease_models'
    label = 'N2'
    N = 12
    day = 'A2'

    #plot_name = '{set}-{l}-{d}.png'.format(set=dataset, l=label, d=day)
    #print plot_name
    #
    #for blob_id in worms:
    #    pick_images_for_worm(blob_id)
    labels = [u'NQ19', u'NQ67', u'MQ0', u'N2', u'MQ40', u'NQ40', u'MQ35']
    for label in labels:
        for day in ['A1', 'A3']:
            for r in ['A', 'B', 'C']:
                json_name = '{set}-{l}-{d}-{r}.json'.format(set=dataset, l=label, d=day, r=r)
                plot_name = '{set}-{l}-{d}-{r}.png'.format(set=dataset, l=label, d=day, r=r)
                try:
                    worms_to_draw =save_drawable_worms(dataset, label, day, N=N, savename=json_name)
                    draw_worms(worms_to_draw, plot_name)
                except Exception as e:
                    print label, day, 'failed'
                    print e
    
