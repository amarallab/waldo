__author__ = 'sallen + peterwinter'
'''
This file contains functions for trying to determine the domains of the worm path where the worm isn't moving.
The main way this is being tested is by using the synthetic_worm_domain_tester function, which imports a json
of synthetic worm data (artificially created by fake_worm_creator.py).
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import scipy.stats as stats
from itertools import izip
import pandas as pd
import copy

TEST_DIR = os.path.dirname(os.path.realpath(__file__)) 
PROJECT_DIR = os.path.abspath(TEST_DIR + '/../../')
SHARED_DIR = PROJECT_DIR + '/code/shared/'
TEST_DATA_DIR = TEST_DIR + '/data/'

sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR + '/code/')

from filtering.filter_utilities import smoothing_function

'''
def new_smooth_xy(times, xy):
    print len(times), len(xy)
    time_threshold = 30
    distance_threshold = 0.25
    
    
    # calculate stationary regions and make positions in them completely stationary.
    point_scores = wd.neighbor_calculation2(distance_threshold=distance_threshold, xy=xy)
    domains = wd.domain_creator(point_scores, timepoint_threshold=time_threshold)
    xy1 = wd.homogenize_domains(xy, domains)
    t, xy2 = zip(*wd.reduce_domains(zip(times,xy1), domains))
    print 'checka', len(xy), len(xy1), len(xy2), len(wd.expand_domains(xy2, domains))

    x, y = zip(*xy2)
    xyt = zip(x, y, t)
    xyt = wd.smoothing_function(xyt, 35, 5)   
    x,y, t = zip(*xyt)
    x, y = list(x), list(y)    
    
    x, y = zip(*xy2)
    x, y = list(x), list(y)
    dt = np.diff(np.array(t))
    dx = np.diff(np.array(x))
    dy = np.diff(np.array(y))
    # to guard against division by zero                                                                                                                                                                                                                                                                                    
    for i, t in enumerate(dt):
        if t < 0.0000001:
            dt[i] = 0.0000001
    print 'check', len(dt), len(dx), len(dy)
    
    speeds = np.sqrt(dx**2 + dy**2) / dt
    v = list(speeds) + [np.nan]
    theta = [np.nan] + list(ac.angle_change_for_xy(x, y, units='rad')) + [np.nan]

    xyvtheta = zip(x,y, v, theta)
    xyvtheta = wd.expand_domains(xyvtheta, domains)
    x, y, v, dtheta = zip(*xyvtheta)
    
    
    # put all data in dataframe form
    new =pd.DataFrame(columns=['x', 'y', 'v', 'dtheta'],
                      index=times)
    new['x'] = x
    new['y'] = y
    new['v'] = v
    new['dtheta'] = dtheta
    return new
'''

def new_smooth_xy(times, xy):
    print len(times), len(xy)
    time_threshold = 30
    distance_threshold = 0.25
    
    
    # calculate stationary regions and make positions in them completely stationary.
    point_scores = wd.neighbor_calculation2(distance_threshold=distance_threshold, xy=xy)
    domains = wd.domain_creator(point_scores, timepoint_threshold=time_threshold)
    xy = wd.homogenize_domains(xy, domains)
    
    
    x, y = zip(*xy)
    xyt = zip(x, y, times)
    xyt = wd.smoothing_function(xyt, 35, 5)   
    x,y, t = zip(*xyt)
    x, y = list(x), list(y)
    
    dt = np.diff(np.array(times))
    dx = np.diff(np.array(x))
    dy = np.diff(np.array(y))
    # to guard against division by zero                                                                                                                                                                                                                                                                                    
    for i, t in enumerate(dt):
        if t < 0.0000001:
            dt[i] = 0.0000001
    print 'check', len(dt), len(dx), len(dy)
    
    
    speeds = np.sqrt(dx**2 + dy**2) / dt
    v = list(speeds) + [np.nan]
    theta = [np.nan] + list(ac.angle_change_for_xy(x, y, units='rad')) + [np.nan]

    xyvtheta = zip(x,y, v, theta)
    print 'check2', len(x), len(y), len(theta), len(v), len(xyvtheta)


    print 'check3', len(xyvtheta), len(times), len(wd.expand_domains(xy, domains))
    x, y, v, dtheta = zip(*xyvtheta)
    new =pd.DataFrame(columns=['x', 'y', 'v', 'dtheta'],
                      index=times)
    new['x'] = x
    new['y'] = y
    new['v'] = v
    new['dtheta'] = dtheta
    return new

def homogenize_domains(xy, domains):
    ''' replace each stationary domain with the median values for that domain '''   
    new_xy = copy.copy(xy)
    for (start, end) in domains:
        end += 1 #because string indicies are non inclusive
        l = end - start
        x_seg, y_seg = zip(*new_xy[start:end])
        x_seg = np.ones(l) * np.median(x_seg)
        y_seg = np.ones(l) * np.median(y_seg) 
        new_xy[start:end] = zip(x_seg, y_seg)
    return new_xy
        
def expand_domains(xy, domains):
    ''' expands the one value left for each domain into multiple values '''
    expansions = {}
    shift = 0
    for start, end in domains:
        #expansions[start - shift] = end - start + 1
        #shift += end - start
        expansions[start - shift] = end - start
        shift += end - start - 1

    #print expansions
    expanded = []
    for i, val in enumerate(xy):
        if i in expansions:
            segment = [val] * expansions[i]
        else:
            segment = [val]
        expanded.extend(segment)
    return expanded

def reduce_domains(xy, domains):
    ''' removes all but one value for each domain '''
    reduction_filter = [True] * len(xy)
    for start, end in domains:
        reduction_filter[start + 1: end] = [False] * (end - start - 1)
    return [v for (f,v) in zip(reduction_filter, xy) if f]

def neighbor_calculation(distance_threshold, xy, time_threshold=500):
    """
    Calculates the score at each point in the worm path, where the score represents the number
    of neighbors within a certain threshold distance.
    :param distance_threshold: This is the threshold distance to be considered a neighbor to a point, assumed here
    to be in mm, not pixels
    :param nxy: A list of xy points, presumed to be noisy, i.e. the recorded data of the worm path
    :param time_threshold: the max number of timepoints that will be tested as neigbors.
    :return: A list of positive integer scores for each point in the worm path.
    """
    point_scores = []
    for i, pt in enumerate(xy[:]):
        start = max([0, i-time_threshold])
        if i < 3:
            point_scores.append(0)
            continue
            
        short_list = xy[start:i]
        x, y = zip(*short_list)
        dx = np.array(x) - pt[0]
        dy = np.array(y) - pt[1]
        dists = np.sqrt(dx**2 + dy**2)
        
        score = 0
        for d in dists[::-1]:
            if d < distance_threshold:
                score += 1
            else:
                break
        #print i, score, dists[0], dists[-1]
        point_scores.append(score)
    return point_scores
    
def domain_creator(point_scores, timepoint_threshold=30):
    """
    Calculates the domains, or regions along the worm path that exhibit zero movement. It does so by setting
    a threshold score for the number of nearby neighbors necessary to be considered an area of no movement, and
    then setting a domain behind that point to be considered non moving.
    
    :param point_scores: The neighbor scores of each point in the worm path
    :param timepoint_threshold: The score threshold for a point to be considered the initiator of a non-moving domain
    :return: A list of domains
    """
    #print 'Calculating Domains'
    threshold_indices = []
    for a, score in enumerate(point_scores):
        if score > timepoint_threshold:
            threshold_indices.append(a)
    domains = []
    for index in threshold_indices:
        score = point_scores[index]
        left = index-score
        if left < 0:
            left = 0
        if domains:
            if domains[-1][1] >= left:
                domains[-1] = [domains[-1][0], index]
            else:
                domains.append([left, index])
        else:
            domains.append([left, index])

    #Takes in the list of domains and merges domains that overlap. This function isn't totally
    #necessary anymore, since I've sort of implemented this logic into the initial calculation

    new_domains = []
    for a, domain in enumerate(domains[:-1]):
        if domain[1] > domains[a+1][0]:
            new_domains.append([domain[0], domains[a+1][1]])
    if not new_domains:
        new_domains = domains

    #print 'Number of domains: {d}'.format(d=len(domains))
    return new_domains

if __name__ == '__main__':
    #a, b, c, d represent the true positive, false positive, true negative, and false negative
    #outputs of the domain_tester function.
    #synthetic_worm_domain_tester(thresholds=[10, 20, 30, 40, 60],
    #                             worm_path='./data/smoothing/synthetic_1.json')
    soln_file = './data/smoothing/paused_soln.h5'
    noise_file = './data/smoothing/noisy_xy_0p1.h5'
    test_bulk_xy(xy_file=noise_file, soln_file=soln_file, r_threshold=1, t_threshold=30)
    plt.show()
    print 'Hello'
