'''
File Name: cluster_pca.py
Author: Peter Winter
Date: 
Description:
'''
# standard imports
import os
import sys

# nonstandard imports
from CURE import *

def write_cluster_file(clusters, savename):
    f = open(savename, 'w')
    for i in clusters:
        #print i
        line = str(i) + '\n'
        f.write(line)
    f.close()

def read_loading_file(loading_file):
    ''' opens loading file and returns a list of lists.     
    '''
    loadings =  []
    for line in open(loading_file, 'r'):
        loadings.append([float(i) for i in line.split()])
    return loadings


def cluster_pca(loading_dir, num_clusters=40):
    '''
    loadings: an 2d np.array or  
    a list in which each entry is a list of numerals

    clusters: a list of cluster ids cooresponding to the loading list
    '''
    assert os.path.exists(loading_dir), 'loading directory not found'
    loading_file = loading_dir + 'loadings.txt'
    assert os.path.exists(loading_file)
    loadings = read_loading_file(loading_file)
    # open cluster instance and generate clusters.
    c = CURE()
    clusters = c.cluster(loadings, num_clusters)    #or: print c.cluster(data, 3, stop_criterion='num_clusters')
    print type(clusters), 'clusters'
    savename = loading_dir + 'cluster_values_cos' +str(num_clusters) + '.txt'
    print 'writing:', savename
    write_cluster_file(clusters, savename)
    # Cluster with distance threshold = 0.001
    #print c.cluster(data, .001, stop_criterion='distance_threshold')


if __name__ == '__main__':
    loading_dir = './PCA-Stats/'
    cluster_pca(loading_dir)

