
# coding: utf-8

# Description:
# This notebook is an attempt to implement the state behaivior model from
# The Geometry of Locomotive Behavioral States in C. elegans
# Gallagher et al.
# 

# In[5]:

# standard imports
import os
import sys
import numpy as np
import scipy.interpolate as interpolate
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import prettyplotlib as ppl

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(os.path.join(HERE, '..'))
SHARED_DIR = os.path.join(CODE_DIR, 'shared')

print CODE_DIR
print SHARED_DIR
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.file_manager import get_timeseries, write_timeseries_file
from centroid import markov_measures_for_xy

def deskew_data(data, e=0.1):
    return np.arcsinh(np.array(data)/e)

def reskew_data(data,e=0.1):
    return np.sinh(np.array(data)) * e

#TODO! this transition matrix initialization does not make sense
def initialize_transition_mat(tau=10, dt=0.1, m=3):
    if m ==1:
        return np.array([1.0])
    off_diag = 1.0 / ((m-1) * tau) * (np.ones((m,m)) - np.identity(m))
    diag = (1 - 1.0/tau) * np.identity(m)
    full = off_diag + diag
    print full
    #return np.exp(full * dt)
    return full

def estimate_params(states, rel_probs, observations, verbose=False):
    all_params = np.zeros((len(states), 7))
    for i, (state, ps) in enumerate(zip(states, rel_probs.T)):
        psum = np.sum(ps)
        wi = ps / psum
        Ci = 1 / (1 - np.sum(wi ** 2))

        
        #pr, mean_s, mean_a, var_s, var_a, var_ar, covar_as = params
        # reversals
        r, s, a, ar = observations[:, 0], observations[:, 1], observations[:, 2], observations[:, 3]
        pr = np.sum(r * wi)
        # speeds
        mean_s = np.sum(s * wi)
        var_s = Ci * (np.sum((s**2) * wi) - mean_s)
        # tangential acceleration
        mean_a = np.sum(a * wi)
        var_a = Ci * (np.sum((a**2) * wi) - mean_a)        
        # radial acceleration
        # mean = 0
        var_ar = np.sum((ar**2) * wi)
        # speed acceleration coovariance
        covar_as = Ci * (np.sum(a * s * wi) - (mean_a * mean_s)) 
        
        if verbose:
            print i, state           
            print '\tpr', pr
            print '\tmean s', mean_s
            print '\tmean_a', mean_a
            print '\tvar s', var_s
            print '\tvar a', var_a
            print '\tvar ar', var_ar
            print '\tcovar as', covar_as
        #for dtype, xi in zip(['r', 's', 'a', 'ar'], observations.T):
        #    print dtype,
        all_params[i, :] = pr, mean_s, mean_a, var_s, var_a, var_ar, covar_as
    return all_params

def plot_param_bars(states, params):

    n = len(states)
    dat = [('probability of reversal','p(r)', 0),
           ('radial acceleration var','variance', -2),
           ('acceleration-speed covariance','covariance a-s', -1)]
    
    fig, axes = plt.subplots(1, len(dat))
    for ax, d in zip(axes, dat):
        title, ylab, i = d
        #ax.set_title(title)
        ax.set_ylabel(ylab)
        ppl.bar(ax, np.arange(n), params[:, i], annotate=True, grid='y')
        ax.set_xticklabels(states)
        
        ax.set_xticks(np.array(range(n)) + 0.5)
    plt.tight_layout()
    plt.show()

def show_param_fits(states, data, rel_probs, mean_list, var_list, bins = 30):

    xlim = [0, int(stats.scoreatpercentile(data, 98) * 1.2)]
    bins = np.linspace(xlim[0], xlim[1], bins)
    
    fig, ax = plt.subplots(2, 1, sharex=True)
    for i, st in enumerate(states):
        state_data = data[rel_probs[:, i] > 0.5]
        ppl.hist(ax[0], state_data, bins = bins, label=st, alpha=0.5, normed=True)
        print i, st, len(state_data)

    for name, m, v in zip(states, mean_list, var_list):
        st = stats.t(v, m)
        x_dummy = np.linspace(xlim[0], xlim[1], 1000)
        ppl.plot(ax[1], x_dummy, st.pdf(x_dummy), lw=2, label=name)
    
    ax[1].set_xlim(xlim) 
    ax[1].legend(loc='upper left')
    return fig, ax

def probablity_of_observation(params, observation):
    '''
    params = [pr, mean_s, mean_a, var_s, var_a, var_ar, covar_as]
    state = [r, s, a, ar]
    '''
    mean_ar = 0 # this is a constant
    # unpack
    pr, mean_s, mean_a, var_s, var_a, var_ar, covar_as = params
    r, s, a, ar = observation
    # set up reversal probablity
    if r:
        Pr = pr
    else:
        Pr = 1 - pr

    # set up all other probabilities
    x = np.array([[s - mean_s], [a - mean_a], [ar - mean_ar]])

    E = np.array([[var_s, covar_as, 0],
                  [covar_as, var_a, 0],
                  [0, 0, var_ar]])

    # calculate all parts of main probability equation
    A = Pr
    B = 4 / ((np.pi ** 2) * np.sqrt(np.linalg.det(E)))
    C = (1 / (1 + np.dot(np.dot(x.T, np.linalg.inv(E)), x))) **3    
    P = A * B * float(C)
    return P

def forward_backward(states, probs, trans_mat, start_probs, end_probs):

    def forward(states, probs, trans_mat, start_probs):
        last_p = start_probs
        forward_p = np.zeros(probs.shape)
        for i, p in enumerate(probs[:]):
            p = p * np.identity(len(states))
            new_p = np.dot(np.dot(p, trans_mat), last_p)            
            new_p = new_p / sum(new_p) # normalize
            forward_p[i] = new_p
            last_p = new_p
        return forward_p

    def backward(states, probs, trans_mat, end_probs):
        probs = probs[::-1,:] # reverse row order
        backward_p = np.zeros(probs.shape)
        last_p = end_probs
        for i, p in enumerate(probs[:]):  
            p = p * np.identity(len(states))
            new_p = np.dot(np.dot(trans_mat, p), last_p) # reverse trans and p from forward algorithm
            new_p = new_p / sum(new_p) # normalize
            backward_p[i] = new_p
            last_p = new_p
        backward_p = backward_p[::-1,:]
        return backward_p           
        
    f = forward(states, probs, trans_mat, start_probs)
    b = backward(states, probs, trans_mat, end_probs)
    posterior = f * b
    # normalize posterior
    for i, p in enumerate(posterior):
        posterior[i] = p / sum(p)
    return posterior

def plot_relative_probability_hists(states, rel_probs, N_bins=30):
    fig, ax = plt.subplots(2, 1)
    for i, (state, ps) in enumerate(zip(states, rel_probs.T)):
        ppl.hist(ax[i], ps, label=state, bins=N_bins, grid='y', normed=True)
        ax[i].legend()
    return fig, ax
    
def plot_relative_probaility_map(states, rel_probs):
    fig, ax = plt.subplots()
    ppl.pcolormesh(fig, ax, rel_probs)
    ax.set_ylabel('time (frames)')
    ax.set_xlabel('states')
    ax.set_xticklabels(states)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.legend()
    return fig, ax



def probablity_of_observation(params, observation):
    '''
    params = [pr, mean_s, mean_a, var_s, var_a, var_ar, covar_as]
    state = [r, s, a, ar]
    '''

    mean_ar = 0 # this is a constant
    # unpack
    pr, mean_s, mean_a, var_s, var_a, var_ar, covar_as = [i for i in params]
    r, s, a, ar = [i for i in observation]
    # set up reversal probablity
    if r:
        Pr = pr
    else:
        Pr = 1 - pr

    # set up all other probabilities
    x = np.array([[s - mean_s], [a - mean_a], [ar - mean_ar]])

    E = np.array([[var_s, covar_as, 0],
                  [covar_as, var_a, 0],
                  [0, 0, var_ar]])

    # calculate all parts of main probability equation
    A = Pr
    B = 4 / ((np.pi ** 2) * np.sqrt(np.linalg.det(E)))
    C = (1 / (1 + np.dot(np.dot(x.T, np.linalg.inv(E)), x))) **3
    
    #print A
    #print B
    #print C
    P = A * B * float(C)
    return P

def forward_backward(states, probs, trans_mat, start_probs, end_probs):
    def forward(states, probs, trans_mat, start_probs):
        last_p = start_probs
        forward_p = np.zeros(probs.shape)
        for i, p in enumerate(probs[:]):
            p = p * np.identity(len(states))
            new_p = np.dot(np.dot(p, trans_mat), last_p)
            new_p = new_p / sum(new_p) # normalize
            forward_p[i] = new_p
            last_p = new_p
        return forward_p

    def backward(states, probs, trans_mat, end_probs):
        probs = probs[::-1,:] # reverse row order
        backward_p = np.zeros(probs.shape)
        last_p = end_probs
        for i, p in enumerate(probs[:]):  
            p = p * np.identity(len(states))
            new_p = np.dot(np.dot(trans_mat, p), last_p) # reverse trans and p from forward algorithm
            new_p = new_p / sum(new_p) # normalize
            backward_p[i] = new_p
            last_p = new_p
            #         if 6970 < i < 6980:
            #             print i, p
            #             print i, last_p
            #             print i, new_p
        backward_p = backward_p[::-1,:]
        return backward_p           
    
    
    f = forward(states, probs, trans_mat, start_probs)
    b = backward(states, probs, trans_mat, end_probs)
    posterior = f * b
    # normalize posterior
    for i, p in enumerate(posterior):
        posterior[i] = p / sum(p)
    return posterior


def closed_loop_fit(states, params, observations, T, start_probs, end_probs, iterations=1, show_plots=False):

    history = []
    for it in range(iterations):
        #print 'itteration ', it
        # calculate probailities each observation 
        # comes from each state completely independent of one another
        probs = np.zeros((len(states), len(observations)))

        # if one state, all points have 100 % chance of being in that state. skip probability calculations. 
        if len(states) == 1:
            rel_probs = np.ones((1, len(observations)))
        else:
            for i, (state, par) in enumerate(zip(states, params)):
                probs[i, :] = [probablity_of_observation(par, obs) for obs in observations]

            # use forward backward algorithm to calculate relative probailities 
            # of each point being in each state with transition pentality
            rel_probs = forward_backward(states, probs.T, T, start_probs, end_probs)

        # recalculate what the parameters should be, given the current split of the data.
        params = estimate_params(states, rel_probs, observations, verbose=False)
        
        # TODO! Turn into weighted average... rather than just standard
        if len(states) > 1:
            for i in range(3,7):
                params[:,i] = np.mean(params[:,i])

        if show_plots:
            # plot speed
            mean_list = [new_params[i][1] for i, _ in enumerate(states)]
            var_list = [new_params[i][3] for i, _ in enumerate(states)]
            fig, ax = show_param_fits(states, data=observations[:,1], rel_probs=rel_probs,
                                      mean_list=mean_list, var_list=var_list, bins=30)
            ax[1].set_xlabel('speed')
            ax[0].set_title('deskewed speed for states')
            plt.show()     

        history.append(params)
        
    return params, rel_probs, history
        
def guess_one_state_params(observations):
    r, s, a, ar = observations[:, 0], observations[:, 1], observations[:, 2], observations[:, 3]
    pr = np.mean(r)
    mean_s = np.mean(s) 
    mean_a = np.mean(a)
    var_s = np.var(s)
    var_a = np.var(a)
    var_ar = np.var(ar)
    cov_as = np.cov(a, s)[1,0]
    return (pr, mean_s, mean_a, var_s, var_a, var_ar, cov_as)


def run_fit(states, params, observations):
    N_states = len(states)
    T = initialize_transition_mat(tau=86400, dt=0.1, m=N_states)
    start_probs = np.ones((N_states,)) * (1.0/N_states)
    end_probs = np.ones((N_states,))
    params, rel_probs, history = closed_loop_fit(states,
                                                 params=params, 
                                                 observations=observations, 
                                                 T=T,
                                                 start_probs=start_probs, 
                                                 end_probs=end_probs, 
                                                 iterations=30)  

    return params, rel_probs, history

def plot_mean_dists(states, params, observations, rel_probs):
    mean_list = [params[i][1] for i, _ in enumerate(states)]
    var_list = [params[i][3] for i, _ in enumerate(states)]
    fig, ax = show_param_fits(states, data=observations[:,1], rel_probs=rel_probs,
                              mean_list=mean_list, var_list=var_list, bins=30)
    ax[1].set_xlabel('speed')
    ax[0].set_title('deskewed speed for states')
    return fig, ax
    
# nonfunctional
def add_parameter_row(params, step=0.5):
    N = 2
    if not isinstance(params, np.ndarray) or not params.shape[1]:
        params = np.ones((1,7)) * params
        print params
    p = np.zeros((N, 7))
    for i in range(params.shape[1]):
        print p[i, :].shape, arams[i: 0].shape
        p[i, :] = params[i: 0]
    p[-1, :] = params[-1, :]
    p[-1, 1] += step
    return p

def fit_hmm_for_blob(blob_id):
    times, xy = get_timeseries(blob_id, data_type='xy')
    x, y = zip(*xy)

    dt = 0.1
    t = times
    observations = markov_measures_for_xy(x, y, dt=0.1)
    #(r, s, a, ar) = observations[:, 0], observations[:, 1], observations[:, 2], observations[:, 3]
    observations[1:, :] = deskew_data(observations[1:, :])

    states = ['one state']
    params = guess_one_state_params(observations)

    states = ['slower', 'faster']
    p = np.zeros((len(states), 7))
    p[0, :] = params
    p[1, :] = params
    p[1, 1] += 1
    params = p
    params, rel_probs, _ = run_fit(states, params, observations)
    plot_mean_dists(states, params, observations, rel_probs)

    states = ['qui', 'dell', 'roam']
    p = np.zeros((len(states), 7))
    p[0, :] = params[0, :]
    p[1, :] = params[1, :]
    p[2, :] = params[1, :]
    p[2, 1] += 1
    params = p
    params, rel_probs,history = run_fit(states, params, observations)
    plot_mean_dists(states, params, observations, rel_probs)
    plot_relative_probaility_map(states, rel_probs)
    #plot_relative_probability_hists(states, rel_probs)

    fig, axes = plt.subplots(7,1, sharex=True)
    for i, ax in enumerate(axes):
        for j, st in enumerate(states):
            dat = [h[j, i] for h in history]
            #print dat
            ppl.plot(ax, dat, label=st)
        ax.legend()
    plt.show()


if __name__ == '__main__':
    blob_id = '20130318_131111_49044'
    fit_hmm_for_blob(blob_id)    
