#!/usr/bin/env python

'''
Filename: spine_processing_videos.py
Description: functions involving the creation of videos to illustrate vairious stages of worm shape processing.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import pylab as pl
import numpy as np
import bisect
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
assert os.path.exists(project_directory), 'project directory not found'
sys.path.append(project_directory)

# nonstandard imports
from flag_timepoints import consolidate_flags
from breaks_and_coils import create_coil_flag_timedict
from create_spine import create_spine_from_outline

from database.mongo_retrieve import pull_data_type_for_blob
from database.mongo_retrieve import timedict_to_list
from Encoding import decode_outline
from GeometricCalculations.matrix_and_point_operations import matrix_to_unordered_points
from filtering.filter_utilities import compute_polynomial_xy_smoothing_by_index
from filtering.filter_utilities import compute_filtered_timedict

def quad_video(rawspine_timedict, raw_outline_timedict, treated_spine_timedict, poly_outline_timedict,
               flag_timedict, coil_timedict, smoothed_spine_timedict):
    coord_shift = 30
    window_size = 60

    pl.ion()
    times = sorted([(float(t.replace('?', '.')), t) for t in treated_spine_timedict])
    smoothed_times = sorted([(float(t.replace('?', '.')), t) for t in smoothed_spine_timedict])
    for i, (tf, t) in enumerate(times):
        stf, st = smoothed_times[bisect.bisect(smoothed_times, times[i]) - 1]
        raw_outline = raw_outline_timedict[t]
        polynomial_outline = poly_outline_timedict[t]
        raw_spine = rawspine_timedict[t]
        treated_spine = treated_spine_timedict[t]
        smoothed_spine = smoothed_spine_timedict[st]
        flag = flag_timedict[t]
        coil = coil_timedict[t]

        ####### plotting starts here ########
        # raw outline + spine raw
        rox = [v[0] - coord_shift for v in raw_outline]
        roy = [v[1] + coord_shift for v in raw_outline]
        pl.plot(rox, roy, color='blue')
        #rsx = [v[0]-coord_shift for v in raw_spine]
        #rsy = [v[1]+coord_shift for v in raw_spine]
        #pl.plot(rsx,rsy, color='green')

        # smoothed outline + treated spine
        px, py = zip(*polynomial_outline)
        pox = [v + coord_shift for v in px]
        poy = [v + coord_shift for v in py]
        tsx = [v[0] + coord_shift for v in treated_spine]
        tsy = [v[1] + coord_shift for v in treated_spine]

        if flag and coil:
            pl.plot(pox, poy, color='blue')
            pl.plot(tsx, tsy, color='green')
        elif not coil:
            pl.plot(pox, poy, color='purple')
            pl.plot(tsx, tsy, color='purple')
        elif not flag:
            pl.plot(pox, poy, color='red')
            pl.plot(tsx, tsy, color='red')

        # if the smoothed spine not ready, use regular.
        center_x = np.mean(rox) + coord_shift
        center_y = np.mean(roy) - coord_shift

        # smoothed in time
        if len(smoothed_spine) > 0:
            #sx = [v[0]+coord_shift for v in smoothed_spine]
            sx = [v[0] for v in smoothed_spine]
            sy = [v[1] - coord_shift for v in smoothed_spine]
            pl.plot(sx, sy, color='green')
            pl.plot([sx[0]], [sy[0]], color='red', marker='o')
            center_x = np.mean(sx)
            center_y = np.mean(sy) + coord_shift

        pl.xlim([int(center_x) - window_size, int(center_x) + window_size])
        pl.ylim([int(center_y) - window_size, int(center_y) + window_size])

        pl.draw()

        #savename = './../Videos/ProcessSpines/process_spine%s_%04i.png' % (blob_id,i)
        #print savename
        #pl.savefig(savename)
        pl.clf()

def duel_video(outline_timedict1={}, spine_timedict1={},
               outline_timedict2={}, spine_timedict2={},
               show_points=True):
    coord_shift = 30
    window_size = 60

    def make_times(timedict1, timedict2):
        mt = lambda x: sorted([(float(t.replace('?', '.')), t) for t in x])
        if timedict1: return mt(timedict1)
        else: return mt(timedict2)



    f1, ax1 = plt.subplots()
    f1.set_size_inches([10,8])
    #frames = [[ax1.plot(*zip(*c), lw=1.5, color='blue')[0],] for c in contours]
    frames = []
    ax1.grid(which='both')
    ax1.set_aspect('equal')

    times1 = make_times(outline_timedict1, spine_timedict1)
    times2 = make_times(outline_timedict2, spine_timedict2)

    for (tf, t) in times1:
        if times1 == times2: t2 = t
        else:
            index = bisect.bisect(times2, (tf,t)) - 1
            tf2, t2 = times2[index]


        def shift_plot(points, shift=[0,0]):
            x, y = zip(*points)
            center = (np.mean(x), np.mean(y))
            x = [i + shift[0] for i in x]
            y = [i + shift[1] for i in y]
            return x, y, center

        ####### plotting starts here ########
        if t in outline_timedict1:
            if outline_timedict1[t]:
                x, y, center = shift_plot(outline_timedict1[t], [-coord_shift, 0])
                frames.append(ax1.plot(x, y, color='blue'))
        '''
        if t in spine_timedict1:
            if spine_timedict1[t]:
                x, y, center = shift_plot(spine_timedict1[t], [-coord_shift, 0])
                frames.append(ax1.plot(x, y, color='blue'))
                if show_points: ax1.plot(x, y, color='blue', marker='.')

        if t2 in outline_timedict2:
            if outline_timedict2[t2]:
                x, y, center = shift_plot(outline_timedict2[t2], [+coord_shift, 0])
                ax1.plot(x, y, color='green')

        if t2 in spine_timedict2:
            if spine_timedict2[t2]:
                x, y, center = shift_plot(spine_timedict2[t2], [+coord_shift, 0])
                ax1.plot(x, y, color='green')
                if show_points: ax1.plot(x, y, color='green', marker='.')
        '''


        #pl.draw()

        #savename = './../Videos/ProcessSpines/process_spine%s_%04i.png' % (blob_id,i)
        #print savename
        #pl.savefig(savename)
        #pl.clf()

    plt.xlim([int(center[0]) - window_size, int(center[0]) + window_size])
    plt.ylim([int(center[1]) - window_size, int(center[1]) + window_size])
    im_ani1 = animation.ArtistAnimation(f1, frames, interval=25, repeat_delay=500, blit=True)
    plt.show()

def spine_construction_video_animationtest(blob_id, save_frames=False, **kwargs):
    from SpineProcessing.Code.skeletonize_outline import compute_skeleton_from_outline
    f1, ax1 = plt.subplots()
    f1.set_size_inches([10,8])

    ax1.grid(which='both')
    ax1.set_aspect('equal')
    
    coord_shift = 30
    window_size = 60

    outline_entry = pull_data_type_for_blob(blob_id, 'encoded_outline', **kwargs)
    encoded_outline_timedict = outline_entry['data']
    raw_outline_timedict, poly_outline_timedict = process_outlines(encoded_outline_timedict)

    #rawspine_entry = pull_data_type_for_blob(blob_id, 'raw_spine')
    #rawspine_timedict = rawspine_entry['data']

    #spine_entry = pull_data_type_for_blob(blob_id, 'treated_spine')
    #treated_spine_timedict = spine_entry['data']


    times = sorted([(float(t.replace('?', '.')), t) for t in raw_outline_timedict])
    #frames = [[ax1.plot(*zip(*c), lw=1.5, color='blue')[0],] for c in contours]
    frames = []
    for i, (tf, t) in enumerate(times):
        d = compute_skeleton_from_outline(raw_outline_timedict[t], return_intermediate_steps=True)
        outline_matrix, filled_matrix, spine_matrix_branched, spine_matrix, corner_pt, endpoints = d

        pt1 = raw_outline_timedict[t][0]
        x1 = pt1[0] - corner_pt[0] #+ 1
        y1 = pt1[1] - corner_pt[1] #+ 1

        ####### plotting starts here ########
        # raw outline points
        outline_points = matrix_to_unordered_points(outline_matrix)
        opx = [v[0] - coord_shift for v in outline_points]
        opy = [v[1] + coord_shift for v in outline_points]
        frames.append(ax1.plot(opx, opy, marker='.', color='black', linewidth=0))
        '''
        plt.plot([x1 - coord_shift], [y1 + coord_shift], marker='.', color='red', linewidth=0)

        ####### filled outline ########
        filled_points = matrix_to_unordered_points(filled_matrix)
        fpx = [v[0] + coord_shift for v in filled_points]
        fpy = [v[1] + coord_shift for v in filled_points]
        plt.plot(fpx, fpy, marker='.', color='black', linewidth=0)
        plt.plot([x1 + coord_shift], [y1 + coord_shift], marker='.', color='red', linewidth=0)

        ####### thinned outline ########
        spine_points = matrix_to_unordered_points(spine_matrix_branched)
        spx = [v[0] - coord_shift for v in spine_points]
        spy = [v[1] - coord_shift for v in spine_points]
        plt.plot(spx, spy, marker='.', color='y', linewidth=0)
        ex = [v[0] - coord_shift for v in endpoints]
        ey = [v[1] - coord_shift for v in endpoints]
        plt.plot(ex, ey, marker='o', color='blue', linewidth=0)

        plt.plot(spx, spy, marker='.', color='y', linewidth=0)

        ####### thinned and filled ########
        fpx = [v[0] + coord_shift for v in filled_points]
        fpy = [v[1] - coord_shift for v in filled_points]
        plt.plot(fpx, fpy, marker='.', color='black', linewidth=0)
        spx = [v[0] + coord_shift for v in spine_points]
        spy = [v[1] - coord_shift for v in spine_points]
        plt.plot(spx, spy, marker='.', color='y', linewidth=0)
        plt.plot([x1 + coord_shift], [y1 - coord_shift], marker='.', color='red', linewidth=0)
        '''

        center_x = np.mean(opx) + coord_shift
        center_y = np.mean(opy) - coord_shift

        plt.xlim([int(center_x) - window_size, int(center_x) + window_size])
        plt.ylim([int(center_y) - window_size, int(center_y) + window_size])

        #pl.draw()
        if save_frames:
            savename = './../Results/spine_matrix%s_%04i.png' % (blob_id, i)
            print savename
        #    pl.savefig(savename)
        #pl.clf()

    print 'showing'
    im_ani1 = animation.ArtistAnimation(f1, frames, interval=25, repeat_delay=500, blit=True)
    plt.show()

        
    
def spine_construction_video_old(blob_id, save_frames=False, **kwargs):
    from SpineProcessing.Code.skeletonize_outline import compute_skeleton_from_outline
    coord_shift = 30
    window_size = 60

    outline_entry = pull_data_type_for_blob(blob_id, 'encoded_outline', **kwargs)
    encoded_outline_timedict = outline_entry['data']
    raw_outline_timedict, poly_outline_timedict = process_outlines(encoded_outline_timedict)


    #rawspine_entry = pull_data_type_for_blob(blob_id, 'raw_spine')
    #rawspine_timedict = rawspine_entry['data']

    #spine_entry = pull_data_type_for_blob(blob_id, 'treated_spine')
    #treated_spine_timedict = spine_entry['data']

    pl.ion()
    times = sorted([(float(t.replace('?', '.')), t) for t in raw_outline_timedict])

    for i, (tf, t) in enumerate(times):
        d = compute_skeleton_from_outline(raw_outline_timedict[t], return_intermediate_steps=True)
        outline_matrix, filled_matrix, spine_matrix_branched, spine_matrix, corner_pt, endpoints = d

        pt1 = raw_outline_timedict[t][0]
        x1 = pt1[0] - corner_pt[0] #+ 1
        y1 = pt1[1] - corner_pt[1] #+ 1

        ####### plotting starts here ########
        # raw outline points
        outline_points = matrix_to_unordered_points(outline_matrix)
        opx = [v[0] - coord_shift for v in outline_points]
        opy = [v[1] + coord_shift for v in outline_points]
        pl.plot(opx, opy, marker='.', color='black', linewidth=0)
        pl.plot([x1 - coord_shift], [y1 + coord_shift], marker='.', color='red', linewidth=0)

        ####### filled outline ########
        filled_points = matrix_to_unordered_points(filled_matrix)
        fpx = [v[0] + coord_shift for v in filled_points]
        fpy = [v[1] + coord_shift for v in filled_points]
        pl.plot(fpx, fpy, marker='.', color='black', linewidth=0)
        pl.plot([x1 + coord_shift], [y1 + coord_shift], marker='.', color='red', linewidth=0)

        ####### thinned outline ########
        spine_points = matrix_to_unordered_points(spine_matrix_branched)
        spx = [v[0] - coord_shift for v in spine_points]
        spy = [v[1] - coord_shift for v in spine_points]
        pl.plot(spx, spy, marker='.', color='y', linewidth=0)
        ex = [v[0] - coord_shift for v in endpoints]
        ey = [v[1] - coord_shift for v in endpoints]
        pl.plot(ex, ey, marker='o', color='blue', linewidth=0)

        pl.plot(spx, spy, marker='.', color='y', linewidth=0)

        ####### thinned and filled ########
        fpx = [v[0] + coord_shift for v in filled_points]
        fpy = [v[1] - coord_shift for v in filled_points]
        pl.plot(fpx, fpy, marker='.', color='black', linewidth=0)
        spx = [v[0] + coord_shift for v in spine_points]
        spy = [v[1] - coord_shift for v in spine_points]
        pl.plot(spx, spy, marker='.', color='y', linewidth=0)
        pl.plot([x1 + coord_shift], [y1 - coord_shift], marker='.', color='red', linewidth=0)


        center_x = np.mean(opx) + coord_shift
        center_y = np.mean(opy) - coord_shift

        pl.xlim([int(center_x) - window_size, int(center_x) + window_size])
        pl.ylim([int(center_y) - window_size, int(center_y) + window_size])

        pl.draw()
        if save_frames:
            savename = './../Results/spine_matrix%s_%04i.png' % (blob_id, i)
            print savename
            pl.savefig(savename)
        pl.clf()


def process_outlines(encoded_outline_timedict):
    outline_timedict = {}
    poly_outline_timedict = {}
    for t in encoded_outline_timedict:
        raw_outline = decode_outline(encoded_outline_timedict[t])
        outline_timedict[t] = raw_outline
        poly_outline_timedict[t] = compute_polynomial_xy_smoothing_by_index(raw_outline,
                                                                            window_size=25,
                                                                            poly_order=4)
    return outline_timedict, poly_outline_timedict


def show_video_for_spine_complete(blob_id, **kwargs):
    outline_entry = pull_data_type_for_blob(blob_id, 'encoded_outline', **kwargs)
    encoded_outline_timedict = outline_entry['data']
    outline_timedict, poly_outline_timedict = process_outlines(encoded_outline_timedict)

    '''
    rawspine_entry = pull_data_type_for_blob(blob_id, 'raw_spine')
    rawspine_timedict = rawspine_entry['data']

    spine_entry = pull_data_type_for_blob(blob_id, 'treated_spine')
    treated_spine_timedict = spine_entry['data']
    '''
    rawspine_timedict, treated_spine_timedict, _ = create_spine_from_outline(blob_id, insert_to_db=False)

    smoothed_spine_entry = pull_data_type_for_blob(blob_id, 'smoothed_spine', **kwargs)
    smoothed_spine_timedict = smoothed_spine_entry['data']

    flags_entry = pull_data_type_for_blob(blob_id, 'flags', **kwargs)
    all_flag_dicts = flags_entry['data']
    flag_timedict = consolidate_flags(all_flag_dicts)
    coil_timedict = create_coil_flag_timedict(all_flag_dicts)

    quad_video(rawspine_timedict, outline_timedict,
               treated_spine_timedict, poly_outline_timedict,
               flag_timedict, coil_timedict,
               smoothed_spine_timedict)

def show_video_for_spine(blob_id, **kwargs):
    outline_entry = pull_data_type_for_blob(blob_id, 'encoded_outline', **kwargs)
    encoded_outline_timedict = outline_entry['data']
    outline_timedict, poly_outline_timedict = process_outlines(encoded_outline_timedict)

    smoothed_spine_entry = pull_data_type_for_blob(blob_id, 'smoothed_spine', **kwargs)
    smoothed_spine_timedict = smoothed_spine_entry['data']

    duel_video(outline_timedict, poly_outline_timedict, smoothed_spine_timedict)                

def spine_smoothing_check(blob_id, **kwargs):
    spine_timedict1 = pull_data_type_for_blob(blob_id, 'smoothed_spine', **kwargs)['data']
    spine_timedict2 = pull_data_type_for_blob(blob_id, 'smoother_spine', **kwargs)['data']
    duel_video(spine_timedict1, {}, spine_timedict2)                

def show_movie(contours):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    f1, ax1 = plt.subplots()
    f1.set_size_inches([10,8])
    frames = [[ax1.plot(*zip(*c), lw=1.5, color='blue')[0],] for c in contours if len(c) > 1]
    ax1.grid(which='both')
    ax1.set_aspect('equal')
    im_ani1 = animation.ArtistAnimation(f1, frames, interval=25, repeat_delay=500, blit=True)
    plt.show()

def path_video(blob_id, **kwargs):
    spine_timedict = pull_data_type_for_blob(blob_id, 'smoothed_spine', **kwargs)['data']
    t1, spines = timedict_to_list(spine_timedict)
    xy_raw_timedict = pull_data_type_for_blob(blob_id, 'xy_raw', **kwargs)['data']
    t2, xy = timedict_to_list(compute_filtered_timedict(xy_raw_timedict))

    f1, ax = plt.subplots()
    f1.set_size_inches([10, 8])

    time_text = ax.text(0.05, 0.9, '')
    line, = ax.plot([], [], lw=2)
    path, = ax.plot(*zip(*xy), lw=1)

    def animate(i):
        thisx, thisy = zip(*spines[i])
        time_text.set_text(str(t1[i]))
        line.set_data(thisx, thisy)
        return line, time_text

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    ani = animation.FuncAnimation(f1, animate, np.arange(1, len(spines)),
                                  interval=25, blit=True, init_func=init)
    ax.grid(which='both')
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    #blob_id = '20130320_164300_02134'
    #blob_id = '00000000_000001_00002'
    #blob_id = '20130320_113528_00379'
    #blob_id = '20130324_115435_04452'
    # show off blob
    blob_id = '20130319_150235_01070'
    blob_id = '20130320_164252_00461'
    #blob_id = '00000000_000001_00003'
    print 'using blob_id', blob_id
    #path_video(blob_id)


    #show_video_for_spine(blob_id)
    #spine_construction_video_animationtest(blob_id)
    #spine_smoothing_check(blob_id)
    #spine_construction_video(blob_id, save_frames=False)
    #show_video_for_spine_complete(blob_id)

    for_show = True
    if for_show:
        show_video_for_spine_complete(blob_id='20130319_150235_01070')
