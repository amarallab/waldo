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
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(sets)+2)]
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


def test_bulk_xy(xy_file, soln_file, t_threshold, r_threshold):
    noisy_xy = pd.read_hdf(xy_file, 'table')
    soln = pd.read_hdf(soln_file, 'table')
    '''
    columns = noisy_xy.columns
    N_trials = len(columns) / 2
    t = noisy_xy.index
    for i in range(N_trials):
        xkey, ykey = ('x'+str(i)), ('y'+str(i))
        x, y  = list(noisy_xy[xkey]), list(noisy_xy[ykey])
    '''
    #point_scores = neighbor_calculation(distance_threshold=r_threshold, nxy=nxy, ppm=ppm)
    #calculated_domains = domain_creator(point_scores, timepoint_threshold=threshold)
    return noisy_xy, soln
    
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

def ROC_plot2(TP, FP, TN, FN, labels):
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
        
if __name__ == '__main__':    
    soln = pd.read_csv('soln.csv', index_col=0)
    noisy = pd.read_csv('noisy.csv', index_col=0)
    smooth = pd.read_csv('smooth.csv', index_col=0)
