import os
import sys
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from pylab import rand, exp
import glob
import csv
import json

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_directory)

DATA_DIR = os.path.abspath('./../Data/')
RESULTS_DIR = os.path.abspath('./../Results/')

#OPTIMIZE = optimize.leastsq
#OPTIMIZE = optimize.leastsq

'''
def exponential(t, a=-10, k=1, c=30):                
    return a * np.log(k+t) + c

t = np.linspace(0, 7, 7)
y = [exponential(i) for i in t]

def exponential_decay(p, x):
    A, K, C = p # amplitude, K, C
    n = np.random.randn(len(x)) #* 0.05 *C
    return A * np.log(K*x) + C + n
'''

def exponential_decay(p, x): 
    A, B, C = p
    x = np.array(x)
    return A * exp(-B * x) + C

def linear(p, x):
    a, b = p
    x = np.array(x)
    return (a * x) + b

def generate_constrained_exponential_decay_function(constrain='C', value=0.0):
    if str(constrain) in ['A', 'a']:
        return lambda p, x: value * exp(- float(p[0]) * x) + float(p[1])
    if str(constrain) in ['B', 'b']:
        return lambda p, x: float(p[0]) * exp(- value * x) + float(p[1])
    if str(constrain) in ['C', 'c']:
        return lambda p, x: float(p[0]) * exp(- float(p[1]) * x) + value


def fit_function_least_squares(x, y, p0=(-10., 1., 50.), fitfunc=exponential_decay):
    errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
    #x = np.array(x)
    #y = np.array(y)
    p1, success = optimize.leastsq(errfunc, p0, args=(x, y))
    return p1, success

def maximize_fit(x, y, p0, fitfunc):
    errfunc = lambda p, x, y: sum(fitfunc(p, x) - y)
    #help(optimize.fmin)
    return optimize.fmin(errfunc, p0, args=(x, y))

def make_fakedata(p_true=(0.5, 40.0, 0.4), num_points=150, fitfunc=exponential_decay, noise=0.01):
    A, K, C = p_true
    x = np.linspace(1., 20., num_points)
    n = np.random.randn(len(x)) * noise * C    
    y = fitfunc(p_true, x) + n
    return x, y, p_true

def plot_raw_vs_fit(x, y, ps, p_labels, fitfunc=exponential_decay):
    fig = plt.figure()

    ax1 = plt.subplot(2,1,1)
    plt.plot(x, y, "ro", label='data')
    time = np.linspace(min(x), max(x), 1000)
    for p, p_label in zip(ps, p_labels):
        ax1 = plt.subplot(2,1,1)
        plt.plot(time, fitfunc(p, time), "-", label=p_label)
        resisuals = np.array(y) -  np.array(fitfunc(p, x))
        ax2 = plt.subplot(2,1,2, sharex=ax1)
        plt.plot(x, resisuals)

    # Legend the plot
    ax1 = plt.subplot(2,1,1)
    plt.xlabel("time [ms]")
    plt.ylabel("displacement [um]")
    plt.legend()


    ax2 = plt.subplot(2,1,2, sharex=ax1)
    plt.plot([x[0], x[-1]], [0, 0], lw=2, color='red')
    plt.xlabel("time [ms]")
    plt.ylabel("residuals")
    plt.show()
    plt.clf()

def get_data(age=1, data_type='centroid_speed'):
    search_path = '{path}/{dt}/age*_combined.txt'.format(path=DATA_DIR, dt=data_type)
    dfiles_by_age = {}
    for dfile in glob.glob(search_path):
        age = dfile.split('/age')[-1].split('_combined')[0]
        dfiles_by_age[int(age)] = dfile
    dfile = dfiles_by_age.get(1, None)
    return [(float(row[0]), map(float, row[1:])) for row in csv.reader(open(dfile, 'r'))]

def rebin_data(t, bins, t_step=10):
    t = sorted(t)
    #print 't', t[0], t[-1], len(t)
    #print int(t[0]), int(t[-1])
    t_new = range(int(t[0]), int(t[-1] + 1), t_step)
    #print 't_new', t_new[0] ,t_new[-1], len(t_new)

    new_bins = [[] for _ in t_new]
    Ns = [0 for _ in t_new]
    for t, b in izip(t, bins):
        bin_num = (int(t) - t_new[0]) / t_step
        #if bin_num == len(t_new):
        #    bin_num = len(t_new) - 1
        #print t, int(t), bin_num, len(t_new)
        new_bins[bin_num] += list(b)
        Ns[bin_num] += len(b)
    return t_new, new_bins, Ns

def fit_goodness(p, x, y, fitfunc):
    errfunc = make_errfunc(fitfunc)
    print 'err', errfunc(p, x, y)


def fit_exponential_decay_fmin(x, y, p0=rand(3), fit_func=exponential_decay):
    # error function to minimize
    e = lambda p, x, y: (abs((fit_func(p,x)-y))).sum()
    # fitting the data with fmin
    p = optimize.fmin(e, p0, args=(x,y))
    print 'estimater parameters: ', p
    return p


def guess_parameters_for_exponential(x, y):
    p0_guesses, labels = [], []

    # guess flat line
    y_mean = np.mean(y)
    p0_guesses.append((y_mean/2, 0.0, y_mean/2))
    labels.append('lin 1')
    p0_guesses.append((1.0, 0.0, y_mean))
    labels.append('lin 2')
    p0_guesses.append((y_mean, 0.0, 0.0))
    labels.append('lin 3')

    # linearize without constant
    y_log = np.log(np.array(y))
    Ap, B = np.polyfit(x, y_log, 1)    
    A = np.exp(Ap)
    p0_guesses.append((A, -B, y_mean))
    labels.append('linearize')
    return p0_guesses, labels

def fit_exponential_decay_robustly(x, y, fp=exponential_decay, p_num=3):
    
    e = lambda p, x, y: (abs((fp(p,x)-y))).sum()

    p_guess, p_labs = guess_parameters_for_exponential(x, y)
    p_guess = [p[:p_num] for p in p_guess]
    best_p, best_label = p_guess[0], p_labs[0]
    lowest_error = e(best_p, np.array(x), np.array(y))

    p_fits, p_labels = [], []
    for p0, label in zip(p_guess, p_labs):
        print p0, 'test'
        err = e(p0, np.array(x), np.array(y))
        print err, 'err'

        p_fmin = fit_exponential_decay_fmin(x, y, p0=p0, fit_func=fp)
        print p_fmin, 'worked'
        err = e(p_fmin, x, y)
        print err, 'err'

        #p_fits.append(p_fmin)
        #p_labels.append(tag)
        if err < lowest_error:
            tag = 'fmin {l}'.format(l=label)
            best_p, best_label, lowest_error = p_fmin, tag, err

        #least squares gets stuck more easily
        p_fit, sucess = fit_function_least_squares(x, y, p0=p0, fitfunc=fp)
        #p_fits.append(p_fit)
        #p_labels.append(tag)
        if err < lowest_error:
            tag = 'lsrs {l}'.format(l=label)
            best_p, best_label, lowest_error = p_fmin, tag, err

    return best_p, best_label, lowest_error


def fit_constrained_decay_in_range(x, y):
    x = np.array(x)
    y = np.array(y)

    params = []
    print np.linspace(0.1, 0.7, num=10)
    best_p, lowest_error = (0.0, 0.0, 0.0), 10000000000.

    p_guesses = [[0.0, 0.0],
                 [1.0, 0.1],
                 [1.0, 0.01]]
    for C in np.linspace(0.0, 0.1, num=10):
        fp = lambda p, x: p[0] * exp(- p[1] * x) + C
        e = lambda p, x, y: (abs((fp(p,x)-y))).sum()
        print 'constrained C=', C
        for p0 in p_guesses:
            p = optimize.fmin(e, p0, args=(x,y))
            err = e(p, x, y)
            p = (p[0], p[1], C)
            print p, err
            params.append(p)
            if err < lowest_error:
                best_p, lowest_error = p, err
    '''
    fp = exponential_decay
    x_fit = np.linspace(min(x), max(x), 1000) 
    plt.plot(x, y, marker='o', lw=0)
    print 'PLOTTING!'
    for p in params:
        print p, len(p)
        plt.plot(x_fit, fp(p, x_fit), "-", color='green')        

    plt.plot(x_fit, fp(best_p, x_fit), lw=3, color='red')
    plt.show()
    '''
    return best_p, '', err

if __name__ == '__main__':
    #from pylab import *    
    # data toggles

    make_data_cache = False
    get_data_from_cache = False
    fitfunc = exponential_decay

    if make_data_cache:
        x, y = zip(*get_data(age=1))
        x, bins = rebin_data(x, y, t_step=10)
        y = [np.mean(b) for b in bins]
        print y
        json.dump({'x':x, 'y':y}, open('fit_test_xy.json', 'w'))

    if get_data_from_cache:
        data = json.load(open('fit_test_xy.json', 'r'))
        x, y = data['x'], data['y']
    else:
        x, y, p_true = make_fakedata(p_true=(10.0, 0.3, 0.5), noise=1)
        print 'real parameters: ', p_true

    '''
    for i in range(20):
        x, y, p_true = make_fakedata(p_true=(i, 0.05, 0.3), noise=0.01)        
        plt.plot(x,y, label=str(p_true))

    plt.legend()
    plt.show()
    '''

    #p, label, err = fit_exponential_decay_robustly(x, y)
    #print p, label, err
    #plot_raw_vs_fit(x, y, [p], [label], fitfunc=fitfunc)
    fit_constrained_decay_in_range(x, y)

