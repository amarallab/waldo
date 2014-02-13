'''
The purpose of this script is to generate all plots needed for a complete understanding of one worm.
1. path
2. timeseries-basic measures
3. timeseries-centroid measures
4. timeseries-spines measures

'''
__author__ = 'peterwinter'


# standard imports
import os
import sys
import random
import numpy as np
import pylab as pl

# path definitions
code_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)
save_dir = os.path.dirname(os.path.realpath(__file__))


# nonstandard imports
from Shared.Code.WormMetrics.spine_measures import compute_spine_measures
from Shared.Code.WormMetrics.centroid_measures import compute_centroid_measures
from Shared.Code.WormMetrics.basic_measures import compute_basic_measures
#from Shared.Code.Settings.data_settings import plotting_locations
from Shared.Code.Database.mongo_retrieve import pull_data_type_for_blob
from Shared.Code.Database.mongo_retrieve import timedict_to_list
from Shared.Code.Encoding.decode_outline import pull_outline

def plot_shape_progression(blob_id):
    outline_timedict = pull_outline(blob_id)
    o_key = random.choice(outline_timedict.keys())
    ox, oy = zip(*outline_timedict[o_key])
    pl.figure()
    pl.plot(ox, oy, color='red')
    pl.axis('equal')
    #pl.savefig(savename +'.png')
    pl.show()

def plot_path(blob_id, savename, **kwargs):
    outline_timedict = pull_outline(blob_id)
    o_key = random.choice(outline_timedict.keys())
    ox, oy = zip(*outline_timedict[o_key])
    source_data_entry = pull_data_type_for_blob(blob_id, 'xy_raw', **kwargs)
    xy_timedict = source_data_entry['data']
    t, xy = timedict_to_list(xy_timedict)
    x, y = zip(*xy)
    pl.figure()
    pl.plot(x, y)
    pl.plot([x[0]], [y[0]], marker='o', color='red')
    pl.plot(ox, oy, color='red')
    pl.axis('equal')
    pl.savefig(savename +'.png')

def plot_outline_shapes(blob_id, savename, dimensions = [4, 4], order='random', title='', **kwargs):
    outline_timedict = pull_outline(blob_id, **kwargs)
    
    spread = 50
    num_shapes = dimensions[0] * dimensions[1]
    pl.figure()
    if order == 'random':
        o_keys = [random.choice(outline_timedict.keys()) for _ in xrange(num_shapes)]
    else:
        skeys = sorted(outline_timedict.keys())
        o_keys = [skeys[(i * len(outline_timedict)) /num_shapes] for i in xrange(num_shapes)]

    for i in xrange(num_shapes):
        o_key = o_keys[i]
        x_, y_ = zip(*outline_timedict[o_key])
        x_shift = np.mean(x_) + (i % dimensions[1]) * spread
        y_shift = np.mean(y_) + (i / dimensions[1]) * spread
        x = [(j - x_shift) for j in x_]
        y = [(j - y_shift) for j in y_]
        #print i, (i / dimensions[1]) * spread, (i % dimensions[1]) * spread
        pl.plot(x, y, color='blue')
    if len(title) > 0:
        pl.title(title)
    pl.axis('equal')
    pl.savefig(savename +'.png')

def plot_timeseries(timeseries_dict, savename):

    num_plots = len(timeseries_dict)
    if num_plots >= 6:
        num_plots = 6
    fig_count = 0
    pl.figure()
    for i, data_type in enumerate(timeseries_dict):
        count = (i+1)%6
        pl.subplot(num_plots, 1 , count)
        t, y = timedict_to_list(timeseries_dict[data_type], remove_skips=True)
        # TODO rather than removing skips, use skips to break up the plot.
        pl.plot(t, y, label=data_type)
        #pl.ylabel(data_type, fontsize='small')
        pl.xlim([min(t), max(t)])
        #pl.yticks(fontsize='small')
        #pl.xticks(fontsize=0)
        pl.legend(loc=2) #, fontsize='small')
        if count == 0:
            #pl.xticks(fontsize='small')
            pl.savefig(savename + str(fig_count) + '.png')
            fig_count += 1
            pl.figure()

    if len(timeseries_dict)%6 !=0:
        #pl.xticks(fontsize='small')
        pl.savefig(savename+ str(fig_count)+'.png')
    else:
        pl.clf()

def single_worm_suite(blob_id, save_dir='./', **kwargs):
    assert os.path.exists(save_dir), 'plotting directory not found' + str(save_dir)

    plot_path(blob_id, savename=save_dir + blob_id + '_1Path', **kwargs)
    plot_outline_shapes(blob_id, savename=save_dir + blob_id + '_2Shapes', **kwargs)

    #timeseries_dict = compute_basic_measures(blob_id)
    #plot_timeseries(timeseries_dict, savename=save_dir + blob_id + '_3Basic')
    timeseries_dict = compute_centroid_measures(blob_id)
    plot_timeseries(timeseries_dict, savename=save_dir + blob_id + '_4Centroids', **kwargs)
    timeseries_dict = compute_spine_measures(blob_id)
    plot_timeseries(timeseries_dict, savename=save_dir + blob_id + '_5Spines', **kwargs)
    #timeseries_dict = compute_eigenworm_measures(blob_id, for_plotting=True)
    #plot_timeseries(timeseries_dict, savename=plotting_dir + blob_id + '_6Eigen')


if __name__ == '__main__':
    # troubled worms
    #blob_id = '20121121_144045_00529'
    #blob_id = '20121116_155532_02746'
    #blob_id = '20121117_184215_00015'
    blob_id = '00000000_000001_00004'
    #blob_id = '20121119_162934_07337'
    #blob_id = '20121118_165046_01818'
    #single_worm_suite(blob_id)
    #savename = 'test'
    #plot_outline_shapes(blob_id, savename, dimensions = [10, 10], order='ordered')
    plot_shape_progression(blob_id=blob_id)