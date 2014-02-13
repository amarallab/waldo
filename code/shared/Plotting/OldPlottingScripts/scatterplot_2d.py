'''
origionally copied from matplotlib gallery
'''

import os
import sys
code_directory =  os.path.dirname(os.path.realpath(__file__)) + '/../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)
#from mongo_scripts.mongo_retrieve import mongo_query
from Shared.Code.Plotting.OldPlottingScripts.scatterplot_2d_preprocesing import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def plot_a_rama(x, y, x_label='', y_label=''):

    nullfmt   = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8,8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labelsaxHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y, alpha=0.2)
    axScatter.xaxis.set_label_text(x_label)
    axScatter.yaxis.set_label_text(y_label)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth
    #axScatter.set_xlim( (-lim, lim) )
    #axScatter.set_ylim( (-lim, lim) )

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )

    plt.show()

if __name__ == '__main__':
    data_type1 = 'age'
    data_type2 = 'stats_size_raw'
    #data_type2 = 'stats_speed_filtered'
    #data_type2 = 'stats_raw_kink_mwt'
    #stat_type2 = '3rd_q'
    stat_type2 = 'median'

    blob_filter = {'strain':'N2', 'growth_medium':'liquid'}

    dataset1_by_blob_id = value_by_blob_id(data_type1, blob_filter)
    dataset2_by_blob_id = stat_by_blob_id(data_type2, stat_type2, blob_filter)

    paired_stats = pair_values_by_blob_id(dataset1_by_blob_id, dataset2_by_blob_id)

    #labels_by_ex_id, datasets_by_ex_id = pull_stats(input_data_type, main_stat)

    #print len(all_ex_ids)
    x_label = str(data_type1) #data_type1.split('stats_')[-1]) + '_' + str(stat_type1)
    y_label = str(data_type2.split('stats_')[-1])  + '_' + str(stat_type2)

    # the random data
    x, y = zip(*paired_stats)
    #x = np.random.randn(1000)
    #y = np.random.randn(1000)
    plot_a_rama(x,y, x_label, y_label)

