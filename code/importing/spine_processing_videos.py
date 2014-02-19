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
HERE = os.path.dirname(os.path.realpath(__file__)) 
PROJECT_DIRECTORY = os.path.abspath(HERE + '/../../')
SHARED_DIRECTORY = PROJECT_DIRECTORY + '/code/shared/'
assert os.path.exists(PROJECT_DIRECTORY), 'project directory not found'
assert os.path.exists(SHARED_DIRECTORY), 'shared code directory not found'
sys.path.append(SHARED_DIRECTORY)

# nonstandard imports
#from flags_and_breaks import consolidate_flags
from wio.file_manager import get_data
from Encoding import decode_outline

def path_video(blob_id, **kwargs):
    t1, spines, _ = get_data(blob_id, data_type='spine')
    t2, xy, _ = get_data(blob_id, data_type='xy_raw')

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
    blob_id = '20130319_150235_00002'
    blob_id = '20130319_150235_00426'
    #blob_id = '00000000_000001_00002'
    print 'using blob_id', blob_id
    path_video(blob_id)


    #show_video_for_spine(blob_id)
    #spine_construction_video_animationtest(blob_id)
    #spine_smoothing_check(blob_id)
    #spine_construction_video(blob_id, save_frames=False)
    #show_video_for_spine_complete(blob_id)

    for_show = False
    if for_show:
        show_video_for_spine_complete(blob_id='20130319_150235_01070')
