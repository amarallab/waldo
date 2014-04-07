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

TEST_DIR = os.path.dirname(os.path.realpath(__file__)) 
PROJECT_DIR = os.path.abspath(TEST_DIR + '/../../')
SHARED_DIR = PROJECT_DIR + '/code/shared/'
TEST_DATA_DIR = TEST_DIR + '/data/'

sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR + '/code/')

from filtering.filter_utilities import smoothing_function
import fake_worm_creator as fwc


def inbounds(list_length, index):
    """
    Simple function for calculating whether a particular index is within the boundaries of a list,
    in this case specifically to make sure that an upcoming point to test doesn't take place before
    the start of the recording, or after the end of the recording.

    :param list_length: The length of the list, as a positive integer
    :param index:The tested index, as an integer
    """
    if index < 0 or index >= list_length:
        return False
    else:
        return True

def domains_to_expansions(domains):
    

def contract_domains(x, y, domains):
    contraction = []
    


def expand_domains(x, y, domains):
    pass


def synthetic_worm_domain_tester(thresholds, worm_path):
    """
    This function is meant to calculate the false positive, false negative, true positive and true negative values
    with regard to our detection of discrete domains where the worm isn't moving much. It varies the distance
    threshold for the neighbor scoring, and calculates how well the domain finder works, along with the distribution
    of the domain lengths.

    :param worm_path: A string of the file path for the synthetic worm data
    :param thresholds: A list of distance thresholds to be tested
    :return: Returns a list of the true positives (tps), false positives (fps), true negatives (tns), and false
    negatives (fns)
    """
    print 'Opening JSON'

    worm = json.load(open(worm_path, 'r'))
    nxy = worm['noisy-xy']
    rxy = worm['raw-xy']
    time = np.array(worm['time'])
    ppm = worm['pixels-per-mm']
    true_domains = worm['domains']
    t_domains_over_score = []

    #Neighbor score of a point necessary to start the creation of a domain
    #Equivalent to minimum number of points of a domain, with points separated
    #by ~0.1 seconds. Thus a score of 30 would be ~3 seconds.
    timepoint_threshold = 30

    #Removing True Domains that are too small to be detected or relevant, as
    #determined by the score threshold for the calculated domains
    c = 0
    for domain in true_domains:
        if domain[1]-domain[0] >= timepoint_threshold:
            t_domains_over_score.append(domain)
        else:
            c += 1
    print 'True Domains under score threshold: {c}'.format(c=c)
    true_domains = t_domains_over_score

    true_domains_timepoints = np.zeros(len(time))

    rx, ry = zip(*rxy)
    rxyt = zip(rx, ry, time)
    tdx, tdy, tdt = domains_to_xyt(rxyt, true_domains)
    
    nx, ny = zip(*nxy)
    nxyt = zip(nx, ny, time)
    sxyt = smoothing_function(nxyt, 11, 5)
    noise_x, noise_y = fwc.noise_calc(nxyt, sxyt)
    print noise_x / float(ppm)

    for domain in true_domains:
        true_domains_timepoints[domain[0]:domain[1]] = np.ones(domain[1]-domain[0]) * 2

    tps = []
    fps = []
    tns = []
    fns = []

    plt.figure(1)
    plt.figure(2)
    plt.figure(3)
    plt.plot(true_bins[1:], true_hist, label='True')

    unmatched_calc_domains_list = []

    for threshold in thresholds:
        point_scores = neighbor_calculation(distance_threshold=0.007, nxy=nxy, ppm=ppm)
        calculated_domains = domain_creator(point_scores, timepoint_threshold=threshold)

        if calculated_domains:
            calculated_domains_lengths = [domain[1]-domain[0] for domain in calculated_domains]
            calc_hist, calc_bins = np.histogram(calculated_domains_lengths, 10, normed=True)
            plt.figure(3)
            plt.plot(calc_bins[1:], calc_hist, label='Threshold {th}'.format(th=threshold))
            cdx, cdy, cdt = domains_to_xyt(nxyt, calculated_domains)

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot(rx, ry, time, label='True Path', alpha=0.7)
            ax.plot(nx, ny, time, label='Noisy Path', alpha=0.7)
            ax.plot(cdx, cdy, cdt, 'cx', label='Calculated Domains', alpha=0.7)
            ax.plot(tdx, tdy, tdt, 'ro', label='True Domains', alpha=0.7)
            plt.title('Threshold {t}'.format(t=threshold))
            unmatched_calc_domains_list.append(percent_unmatched_calculated_domains(calculated_domains, true_domains))
        else:
            unmatched_calc_domains_list.append(0)
        tp, fp, tn, fn = confusion_matrix(calculated_domains, true_domains_timepoints)
        tps.append(tp)
        fps.append(fp)
        tns.append(tn)
        fns.append(fn)

        calc_per_true_domains, percent_covered_true = calculated_domains_per_true_domain(calculated_domains,
                                                                                         true_domains)
        ctot_hist, ctot_bins = np.histogram(calc_per_true_domains, bins=range(0, 6))
        pcovt_hist, pcovt_bins = np.histogram(percent_covered_true)

def calc_domains_for_xy(x, y, distance_threshold=0.007, timepoint_threshold=30):

    """
    """

    nxy = zip(x,y)
    ppm = 40
    #nxy = worm['noisy-xy']
    #time = np.array(worm['time'])

    #Neighbor score of a point necessary to start the creation of a domain
    #Equivalent to minimum number of points of a domain, with points separated
    #by ~0.1 seconds. Thus a score of 30 would be ~3 seconds.
    
    point_scores = neighbor_calculation(distance_threshold=distance_threshold,
                                        nxy=nxy, ppm=ppm)
    calculated_domains = domain_creator(point_scores,
                                        timepoint_threshold=timepoint_threshold)
    return calculated_domains
    

def get_true_domains(velocities):
    domains = [] 
    in_domain = False
    domain_start = 0
    for a, v in enumerate(velocities):
        if in_domain and v != 0:
            in_domain = False
            domain_end = a-1
            domains.append([domain_start, domain_end])
        elif not in_domain and v == 0:
            in_domain = True
            domain_start = a
    return domains

def plot_confusion_matrix(tps, fps, tns, fns, thresholds):
    """
    Plots the true positive, true negative, false positive, and false negative counts
    vs the distance thresholds they were calculated over.
    :param tps: A list of true positive scores
    :param fps: A list of false positive scores
    :param tns: A list of true negative scores
    :param fns: A list of false negative scores
    :param thresholds: A list of distance thresholds
    """
    plt.figure()
    plt.plot(thresholds, tps, label='True Positives')
    plt.plot(thresholds, fps, label='False Positives')
    plt.plot(thresholds, tns, label='True Negatives')
    plt.plot(thresholds, fns, label='False Negatives')
    plt.ylabel('Count (Number of Points)')
    plt.xlabel('Distance Threshold (mm)')
    plt.legend()


def percent_unmatched_calculated_domains(c_domains, t_domains):
    """
    This calculates the percentage of calculated domains that were not
    :param c_domains:
    :param t_domains:
    :return:
    """
    unmatched_calc_domains = 0
    length_matched = []
    length_unmatched = []
    for domain in c_domains:
            counter = 0
            for td in t_domains:
                if domain[0] > td[1] or domain[1] < td[0]:
                    continue
                else:
                    counter += 1
            if counter == 0:
                unmatched_calc_domains += 1
                length_unmatched.append(domain[1] - domain[0])
            else:
                length_matched.append(domain[1] - domain[0])
    percent_unmatched = unmatched_calc_domains / float(len(c_domains)) * 100
    print stats.ks_2samp(length_matched, length_unmatched)
    return percent_unmatched


def confusion_matrix(c_domains, t_domains_timepoints):
    """
    Calculates the true positive, true negative, false positive, and false negative
    points in the calculated domains by comparing them to the true domain points. It does so
    by first calculating the timepoints within the calculated domains and assigning them a value
    of 1. The timepoints within a true domain have a value of 2. Both lists are of the same length
    as the total number of timepoints in the worm path, meaning some indices have a value of zero.
    When these two lists are subtracted from one another, it can result in four different values:
    -1, 0, 1, or 2. These correspond logically to a false positive, true negative, true positive, and
    false negative, respectively.
    :param c_domains: A list of the calculated domains
    :param t_domains_timepoints: A list of the timepoints within true domains
    :return: The true positive count, false positive count, true negative count, and false negative count.
    """
    calculated_domains_timepoints = np.zeros(len(t_domains_timepoints))
    for domain in c_domains:
            calculated_domains_timepoints[domain[0]:domain[1]] = np.ones(domain[1]-domain[0]) * 1
    results = t_domains_timepoints - calculated_domains_timepoints
    tp = results.tolist().count(1)
    fp = results.tolist().count(-1)
    tn = results.tolist().count(0)
    fn = results.tolist().count(2)
    return tp, fp, tn, fn


def calculated_domains_per_true_domain(c_domains, t_domains):
    """
    For every true domain, this counts the number of calculated domains that overlaps it.
    :param c_domains: A list of the calculated domains
    :param t_domains: A list of the true domains
    :return: A list of positive integers of the same length as the number of true domains.
    """
    calc_per_true_domains = np.zeros(len(t_domains))
    percent_covered_true = np.zeros(len(t_domains))
    for a, td in enumerate(t_domains):
            if td[1] - td[0] != 0:
                for b, cd in enumerate(c_domains):
                    if cd[0] > td[1]:
                        break
                    elif cd[1] < td[0]:
                        continue
                    else:
                        calc_per_true_domains[a] += 1
                        if cd[0] > td[0] and cd[1] < td[1]:
                            percent_covered_true[a] += cd[1] - cd[0]
                        elif cd[0] < td[0] and cd[1] > td[1]:
                            percent_covered_true[a] += td[1] - td[0]
                        elif cd[0] > td[0]:
                            percent_covered_true[a] += td[1] - cd[0]
                        elif cd[1] < td[1]:
                            percent_covered_true[a] += cd[1] - td[0]
                percent_covered_true[a] /= float(td[1] - td[0]) / 100
    return calc_per_true_domains, percent_covered_true


def calc_distance(point1, point2, scalar):
    """
    Calculates the euclidean distance between two points, and divides by a scalar, if necessary,
    to convert units (e.g. from pixels to mm).
    :param point1: X,Y coordinate of the first point
    :param point2: X,Y coordinate of the second point
    :param scalar: Here, a conversion factor in pixels-per-mm
    :return: A float, the distance.
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    vec = np.subtract(p2, p1)
    distance = np.linalg.norm(vec) / float(scalar)
    return distance


def neighbor_calculation(distance_threshold, nxy, ppm=40):
    """
    Calculates the score at each point in the worm path, where the score represents the number
    of neighbors within a certain threshold distance.
    :param distance_threshold: This is the threshold distance to be considered a neighbor to a point, assumed here
    to be in mm, not pixels
    :param nxy: A list of xy points, presumed to be noisy, i.e. the recorded data of the worm path
    :param ppm: A conversion factor of pixels-per-mm for converting the distance between points from pixels to mm
    :return: A list of positive integer scores for each point in the worm path.
    """
    worm_point_scores = []
    print 'Peak distance: {p}'.format(p=distance_threshold)
    print 'Calculating Peaks'
    for a, point in enumerate(nxy):
        flag1 = True
        b = 1
        point_score = 0
        while flag1:
            if inbounds(len(nxy), a-b):
                if calc_distance(point, np.array(nxy[a-b]), ppm) <= distance_threshold:
                    point_score += 1
                else:
                    flag1 = False
            else:
                flag1 = False
            b += 1
        worm_point_scores.append(point_score)
    return worm_point_scores


def plot_domains(domains, y):
    """
    A simple function for graphing a visual representation of the domains of zero movement in the worm
    path. Graphs the domains as a horizontal line at a particular y value.
    :param domains: A list of domains, where domains are themselves a list of two values representing the
    time indices framing a period of zero movement in the worm path.
    :param y: A y value at which the function will graph the horizontal line.
    :return: []
    """
    print 'Plotting Domains'
    for domain in domains:
        plt.hlines(y, domain[0], domain[1])
    return []


def domains_to_xyt(xyt, domains):
    """
    This function takes in xyt data and a list of domains and converts the list of domains into a list of
    xyt points within those domains. This list can then be used to graph the domains onto the entire worm
    path, for visualization.
    :param xyt: A list of xyt points.
    :param domains: A list of domains, which are themselves a list of two values, representing time indices
    that frame a period of zero movement in the worm path.
    :return: Three lists, each one representing values of x, y, and t within the given input domains. These can
    be zipped together to get a list of xyt points within the domains.
    """
    x, y, t = zip(*xyt)
    domains_x = []
    domains_y = []
    domains_t = []
    for domain in domains:
        left = domain[0]
        right = domain[1]
        domains_x.extend(x[left:right])
        domains_y.extend(y[left:right])
        domains_t.extend(t[left:right])
    return domains_x, domains_y, domains_t


def domain_creator(point_scores, timepoint_threshold=30):
    """
    Calculates the domains, or regions along the worm path that exhibit zero movement. It does so by setting
    a threshold score for the number of nearby neighbors necessary to be considered an area of no movement, and
    then setting a domain behind that point to be considered non moving.
    
    :param point_scores: The neighbor scores of each point in the worm path
    :param timepoint_threshold: The score threshold for a point to be considered the initiator of a non-moving domain
    :return: A list of domains
    """
    print 'Calculating Domains'
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
            if domains[-1][1] > left:
                domains[-1] = [domains[-1][0], index]
            else:
                domains.append([left, index])
        else:
            domains.append([left, index])
    new_domains = domain_merge(domains)
    print 'Number of domains: {d}'.format(d=len(domains))
    return new_domains


def ROC_plot(TP, FP, TN, FN):
    """
    Calculates the false positive rate and the true positive rate and plots
    the two against one another with FPR on the x axis and TPR on the y axis.
    :param TP: A list of true positive counts
    :param FP: A list of false positive counts
    :param TN: A list of true negative counts
    :param FN: A list of false negative counts
    :return: []
    """
    TPR = []
    FPR = []
    precision = []
    F1_score = []
    plt.figure()
    for a, tp in enumerate(TP):
        tpr = tp/float(tp+FN[a])
        fpr = FP[a]/float(FP[a]+TN[a])
        TPR.append(tpr)
        FPR.append(fpr)
        precision.append(tp/float(tp+FP[a]))
        F1_score.append((2*tp)/float(2*tp + FP[a] + FN[a]))
        plt.annotate('{index}'.format(index=a), xy=(fpr, tpr), xytext=(fpr, tpr-0.01))
    print TPR, FPR, precision, F1_score
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '-r')
    plt.scatter(FPR, TPR)
    plt.xlabel('False Positive Rate FP/(FP+TN)')
    plt.ylabel('True Positive Rate TP/(TP+FN)')
    return []


def domain_merge(domains):
    """
    Takes in the list of domains and merges domains that overlap. This function isn't totally
    necessary anymore, since I've sort of implemented this logic into the initial calculation
    of the domains.
    :param domains: A list of domains, which are themselves lists with two numbers, the start and
    stop indices of a period of zero movement in the worm path.
    :return: A list of domains, of equal or smaller size of the input domain list.
    """
    new_domains = []
    for a, domain in enumerate(domains[:-1]):
        if domain[1] > domains[a+1][0]:
            new_domains.append([domain[0], domains[a+1][1]])
    if not new_domains:
        new_domains = domains
    return new_domains

if __name__ == '__main__':
    #a, b, c, d represent the true positive, false positive, true negative, and false negative
    #outputs of the domain_tester function.
    synthetic_worm_domain_tester(thresholds=[10, 20, 30, 40, 60],
                                 worm_path='./data/smoothing/synthetic_1.json')
    plt.show()
    print 'Hello'
