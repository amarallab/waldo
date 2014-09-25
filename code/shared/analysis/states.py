# coding: utf-8

# Description:
# This notebook is an attempt to implement the state behaivior model from
# The Geometry of Locomotive Behavioral States in C. elegans
# Gallagher et al.
#

### Imports

# In[2]:

# standard imports
import os
import sys
import numpy as np
import scipy
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
from wio.file_manager import get_timeseries, write_timeseries_file, write_metadata_file

### Functions

def deskew_data(data, e=0.1):
    return np.arcsinh(np.array(data)/e)

def reskew_data(data,e=0.1):
    return np.sinh(np.array(data)) * e

#### Calculation Functions
def markov_measures_for_xy(x, y, dt=0.1, verbose=False):
    """

    reversal
    speed
    acceleration
    angular acceleration

    """
    vs = np.zeros((2, len(x)-1))

    vs[0] = np.diff(x) / dt
    vs[1] = np.diff(y) / dt
    vs = vs.T

    data = []
    v23 = vs[0]
    for v in vs[1:]:
        v12, v23 = v23, v

        if np.dot(v12, v23) < 0:
            r = 1
            d = (v12 - v23) / np.linalg.norm(v12 - v23)
            alpha = (v23 + v12) / dt
            R = [[d[0], d[1]], [-d[1], d[0]]]
        else:
            r = 0
            d = (v12 + v23) / np.linalg.norm(v12 + v23)
            alpha = (v23 - v12) / dt
            R = [[-d[0], -d[1]], [-d[1], d[0]]]

        s = (np.linalg.norm(v12) + np.linalg.norm(v23)) / dt
        if s == 0:
            a, ar = 0, 0
        else:
            a = np.dot(R, alpha)
            ar = a[1]
            a = np.linalg.norm(a)

        if verbose:
            print 'r={r} | s={s} | a={a} | ar={ar}'.format(r=r, s=s, a=a, ar=ar)
        if np.isnan(a):
            print 'nan'

        data.append((r, s, a, ar))

    data = np.array(data)

    for i,dat in enumerate(data):
        if any(np.isnan(dat)):
            print i, dat
    return data

# In[8]:

def initialize_transition_mat(tau=10, dt=0.1, m=3):
    if m ==1:
        return np.array([1.0])
    off_diag = 1.0 / ((m-1) * tau) * (np.ones((m,m)) - np.identity(m))
    diag = ( - 1.0/tau) * np.identity(m)
    full = off_diag + diag
    return scipy.linalg.expm(full*dt)
    return full


# In[9]:

def estimate_params(N_states, rel_probs, observations, verbose=False):
    all_params = np.zeros((N_states, 7))
    for i, ps in enumerate(rel_probs.T):
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
            print i
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


# In[11]:

def probablity_of_observation(params, observation):
    '''
    params = [pr, mean_s, mean_a, var_s, var_a, var_ar, covar_as]
    state = [r, s, a, ar]
    '''
    mean_ar = 0 # this is a constant
    # unpack
    pr, mean_s, mean_a, var_s, var_a, var_ar, covar_as = list(params)
    r, s, a, ar = list(observation)
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

def calculate_state_probabilities_no_memory(N_states, params, observations):
    probs = np.zeros((len(observations), N_states))
    for i, par in enumerate(params):
        probs[:, i] = [probablity_of_observation(par, obs) for obs in observations]
    return probs


# In[15]:

def baum_welch(probs, T, initial=None):
    N_obs, N_states = probs.shape
    assert (N_states, N_states) == T.shape, 'rel_probs and Transition matrix do not have matching numbers of states'

    print N_states ** 2, 'transitions'
    all_transition_points = np.ones(shape=(N_obs-1, N_states**2))


    if initial == None:
        initial = (1.0 / N_states) * np.ones(N_states)

    p = probs[1, :]
    for i, p_new in enumerate(probs[1:,:]):
        p, p_old = p_new, p # step forward
        if p_old == None: # skip first value.
            continue
        current = np.array(T) * p
        current = (current.T * p_old * initial).T
        all_transition_points[i,:] = current.flatten()

    highest_prob = np.amax(all_transition_points, axis=1).sum()
    new_T = (all_transition_points.sum(axis=0) / highest_prob).reshape((N_states, N_states))
    row_sums = new_T.sum(axis=1)
    new_T = new_T / row_sums[:, np.newaxis]
    return new_T

def test_baum_welch():
    # data and solution
    T = np.array([[0.5, 0.5], [0.3, 0.7]])
    p = np.array([[.3, .8], [.3, .8], [.3, .8], [.3, .8], [.3, .8],
                  [.7, .2], [.7, .2], [.3, .8], [.3, .8], [.3, .8]])
    initial = np.array([0.2, 0.8])
    solution_T = np.array([[0.39726027,  0.60273973],
                           [ 0.18333333,  0.81666667]])
    T2 = baum_welch(p, T, initial)
    print 'solution'
    print solution_T
    print 'T'
    print T2
    print 'difference'
    print solution_T - T2

# TODO: Get baum_welch alogrithm working for re-estimating the transition matrix for real data
'''
print states
print params.shape
print observations.shape
probs = calculate_state_probabilities_no_memory(states, params, observations)
print probs.shape
print baum_welch(probs, T)
'''


# In[12]:

def forward_backward(N_states, probs, trans_mat, start_probs, end_probs):

    def forward(N_states, probs, trans_mat, start_probs):
        last_p = start_probs
        forward_p = np.zeros(probs.shape)
        for i, p in enumerate(probs[:]):
            p = p * np.identity(N_states)
            new_p = np.dot(np.dot(p, trans_mat), last_p)
            new_p = new_p / sum(new_p) # normalize
            forward_p[i] = new_p
            last_p = new_p
        return forward_p

    def backward(N_states, probs, trans_mat, end_probs):
        probs = probs[::-1,:] # reverse row order
        backward_p = np.zeros(probs.shape)
        last_p = end_probs
        for i, p in enumerate(probs[:]):
            p = p * np.identity(N_states)
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


    f = forward(N_states, probs, trans_mat, start_probs)
    b = backward(N_states, probs, trans_mat, end_probs)
    posterior = f * b
    # normalize posterior
    for i, p in enumerate(posterior):
        posterior[i] = p / sum(p)
    return posterior


# In[56]:

def closed_loop_fit(params, observations, T, start_probs, end_probs, max_iterations=1, sim_threshold=0.1):

    N_states, _ = T.shape
    history = []
    for it in range(max_iterations):
        #print 'itteration ', it
        # calculate probailities each observation
        # comes from each state completely independent of one another
        probs = calculate_state_probabilities_no_memory(N_states, params, observations)

        # use forward backward algorithm to calculate relative probailities
        # of each point being in each state with transition pentality
        rel_probs = forward_backward(N_states, probs, T, start_probs, end_probs)

        # recalculate what the parameters should be, given the current split of the data.
        params = estimate_params(N_states, rel_probs, observations, verbose=False)

        # TODO! Turn into weighted average... rather than just standard
        #         if len(states) > 1:
        #             for i in range(3,7):
        #                 params[:,i] = np.mean(params[:,i])
        weights = rel_probs[:, :].sum(axis=0) / rel_probs.sum()
        for i in range(3,7):
            params[:,i] = np.average(params[:,i], weights=weights)

        if it >= 1:
            diff = history[-1] - params
            if diff.sum().sum() <= 0.1:
                break
        history.append(params)

    print it, 'itterations'
    # estimate params one final time with variances unconstrained.
    params = estimate_params(N_states, rel_probs, observations, verbose=False)
    history.append(params)

    # run baum-welch to estimate T now that we have decent state params.
    probs = calculate_state_probabilities_no_memory(N_states, params, observations)
    T = baum_welch(probs, T)
    return params, T, rel_probs, history

# TODO: This is still nonfunctional.
def add_parameter_row(params, step=0.5):
    N = 2
    if not isinstance(params, np.ndarray) or not params.shape[1]:
        params = np.ones((1,7)) * params
        print params
    p = np.zeros((N, 7))
    for i in range(params.shape[1]):
        print p[i, :].shape, params[i: 0].shape
        p[i, :] = params[i: 0]
    p[-1, :] = params[-1, :]
    p[-1, 1] += step
    return p

def excess_entropy(rel_probs):
    N, M = rel_probs.shape
    A = sum([(rel_probs[:,i] * np.log2(rel_probs[:,i])).sum() for i in range(M)]) / N
    pi = rel_probs.sum(axis=0) / N
    B = (pi * np.log2(pi)).sum()
    return A - B


def fit_one_state(observations):
    N_states = 1
    T = initialize_transition_mat(tau=1, dt=0.1, m=N_states)
    rel_probs = np.ones((len(observations), 1), dtype=float)
    params = estimate_params(N_states, rel_probs, observations, verbose=False)
    history = []
    return params, T, rel_probs, history

def fit_two_states(params1, observations, tau, dt=0.1):

    N_states = 2

    p = np.zeros((N_states, 7))
    p[0, :] = p[1, :] =params1
    p[1, 1] += 1
    params = p

    T = initialize_transition_mat(tau=tau, dt=0.1, m=N_states)
    start_probs = np.ones((N_states,)) * (1.0/N_states)
    end_probs = np.ones((N_states,))
    params, T, rel_probs, history = closed_loop_fit(params=params,
                                                    observations=observations,
                                                    T=T,
                                                    start_probs=start_probs,
                                                    end_probs=end_probs,
                                                    max_iterations=30)
    return params, T, rel_probs, history

def fit_three_states(params2, observations, tau, dt=0.1):
    N_states = 3
    p = np.zeros((N_states, 7))
    p[0, :] = params2[0, :]
    p[1, :] = p[2, :] = params2[1, :]
    p[2, 1] += 1
    params = p

    T = initialize_transition_mat(tau=tau, dt=0.1, m=N_states)
    start_probs = np.ones((N_states,)) * (1.0/N_states)
    end_probs = np.ones((N_states,))
    params, T, rel_probs, history = closed_loop_fit(params=params,
                                                    observations=observations,
                                                    T=T,
                                                    start_probs=start_probs,
                                                    end_probs=end_probs,
                                                    max_iterations=30)
    return params, T, rel_probs, history



def initialize_data(blob_id):

    times, xy = get_timeseries(blob_id, data_type='xy')
    x, y = zip(*xy)
    observations = markov_measures_for_xy(x, y, dt=0.1)
    ds_observations = np.array(observations)
    ds_observations[1:, :] = deskew_data(observations[1:, :])
    return times, xy, ds_observations, observations


def fit_hmm_for_blob(blob_id, tau=20000, dt=0.1):

    times, xy, ds_observations, observations = initialize_data(blob_id)
    params1, T1, rel_probs1, history1 = fit_one_state(ds_observations)
    entropy1 = excess_entropy(rel_probs1)

    data = {'T':[], 'entropy':entropy1, 'params':params1.tolist()}
    write_metadata_file(blob_id, data_type='markov_states1', data=data)

    params2, T2, rel_probs2, history2 = fit_two_states(params1, ds_observations, tau=tau)
    entropy2 = excess_entropy(rel_probs2)

    write_timeseries_file(blob_id, data_type='markov_prob2', times=times, data=rel_probs2)
    data = {'T':T2.tolist(), 'entropy':entropy2, 'params':params2.tolist()}
    write_metadata_file(blob_id, data_type='markov_states2', data=data)

    params3, T3, rel_probs3, history3 = fit_three_states(params2, ds_observations, tau=tau)
    entropy3 = excess_entropy(rel_probs3)

    write_timeseries_file(blob_id, data_type='markov_prob3', times=times, data=rel_probs3)
    data = {'T':T3.tolist(), 'entropy':entropy3, 'params':params3.tolist()}
    write_metadata_file(blob_id, data_type='markov_states3', data=data)
    print 'hhm fitting finished!'

if __name__ == '__main__':
    blob_id = '20130318_131111_49044'
    fit_hmm_for_blob(blob_id)
