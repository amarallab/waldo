import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from itertools import izip

def compare_dfs(soln, test_dfs=[], labels=[], cols=None, N_bins=None):
    '''
    given a list of result datasets,this computes
    a histogram for all columns 
    compares columns of all dataframes test_dfs against
    
    creates a histogram of the residuals.

    soln: (dataframe)
        the dataframe with the correct answers
    soln: (dataframe)
        the dataframe with the correct answers
        
    '''            

    if N_bins == None:
        N_bins = len(soln)/30
    if cols == None:
        cols = list(soln.columns)
    if labels == []:
        labels = [str(i) for i, _ in enumerate(test_dfs)]

    # creates a list of dataframes containing residuals
    res_list = [(df - soln) for df in test_dfs]
                
    # if scaling all datasets to same hist bins, make sure
    col_bins = {}
    # make 
    def filter_residual(res):
        data = res[col][res[col].notnull()]
        if len(data) == 0:
            data = [0.0]
        return data
    
    for col in cols:
        global_mx = max([max(filter_residual(res)) for res in res_list])
        global_mn = min([min(filter_residual(res)) for res in res_list])        
        #global_mx = max([max(res[col][res[col].notnull()]) for res in res_list])
        #global_mn = min([min(res[col][res[col].notnull()]) for res in res_list])
        print col, global_mn, global_mx        
        col_bins[col] = np.linspace(global_mn, global_mx, N_bins)    
    
    hists = {}
    for label, res in izip(labels, res_list):
        hists[label] = {}
        for col in cols:                
            bins = col_bins[col]
            data = res[col][res[col].notnull()]         
            y, borders = np.histogram(data, bins=bins)
            centers = (borders[1:] + borders[:-1]) / 2.0
            hists[label][col] = zip(centers, y)
    return hists

# great for debugging.
'''
r = 135, 140
print r
n = noisy[r[0]:r[1]]
n.columns = ['n-' + c for c in n.columns]
s = soln[r[0]:r[1]]
print pd.concat([n, s], axis=1)
'''
        
def plot_smoothing_explorer(soln, sets=[], labels=[], columns=None, colors=None):
    # toggles
    narrow_window = False
    #narrow_window = True
    
    if columns == None:
        columns = list(soln.columns)
    N_col = len(columns)

    # TODO set up color rotiation
    if colors == None:
        colormap = cm.spectral
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(sets)+1)]
        sol_color = colors[-1]
    
    # for each dataframe in sets, calculate the residule dist
    residual_hists = compare_dfs(soln, test_dfs=sets, labels=labels)    
    
    # set up all axis alignment
    gs = gridspec.GridSpec(N_col, 4)
    ax1 = plt.subplot(gs[0, :-1])
    time_ax = [plt.subplot(gs[i, :-1], sharex=ax1) for i in range(1, N_col)]
    time_ax = [ax1] + time_ax
    res_ax = [plt.subplot(gs[i, -1]) for i in range(N_col)]

                      
    # plot all timeseries putting each col in a seperate plot.
    for i, col in enumerate(columns):
        time_ax[i].plot(soln[col], label='soln', lw=2, color=sol_color)
        for label, color, dframe in izip(labels, colors, sets):
            # plot timeseries            
            time_ax[i].plot(dframe[col], label=label, alpha=0.5, color=color)

            # plot histogram
            x, y = zip(*residual_hists[label][col])
            res_ax[i].plot(x, y, label=label, alpha=0.5, color=color)
            res_ax[i].fill_between(x, y, label=label, alpha=0.1, color=color)
        time_ax[i].set_ylabel(col)
        #time_ax[i].tick_params(axis='x', color='white')

    time_ax[0].legend()
        
    if narrow_window:
        time_ax[0].set_xlim([r[0], r[1]])


        
if __name__ == '__main__':    
    soln = pd.read_csv('soln.csv', index_col=0)
    noisy = pd.read_csv('noisy.csv', index_col=0)
    smooth = pd.read_csv('smooth.csv', index_col=0)
