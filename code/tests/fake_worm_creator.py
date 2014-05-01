__author__ = 'sallen'
'''
This file contains functions that will generate synthetic/fake worm centroid xyt path data. The functions can
be called from another file, and will return a dictionary of several sets of points for a given worm. If run
from within this file, it will save the worm data as a json file that can be accessed from another file later.
'''
# standard imports
import os
import sys
import json
import glob
import numpy as np
import scipy.stats as sts
import random
import matplotlib.pyplot as plt
import pandas as pd

TEST_DIR = os.path.dirname(os.path.realpath(__file__)) 
PROJECT_DIR = os.path.abspath(TEST_DIR + '/../../')
CODE_DIR = os.path.join(PROJECT_DIR, 'code')
SHARED_DIR = os.path.join(CODE_DIR, 'shared')
TEST_DATA_DIR = TEST_DIR + '/data/'

sys.path.append(SHARED_DIR)
sys.path.append(CODE_DIR)

# nonstandard imports
import filtering.filter_utilities as fu
from metrics.compute_metrics import rescale_radians, angle_change_for_xy

def generate_individual_series(N_points=3600):
    """
    Generates an entire individual timeseries
    """
    # toggles
    reorientation_chance = 10
    
    state = (0, 0, 0, 0)
    states = [state]    

    while len(states) < N_points + 2:
        N_seg = int(sts.expon.rvs(scale=100))

        #intialize velocity and angular change
        new_v = random.choice([0, 0.01, 0.04])
        d_ang = 0
        if new_v > 0:
            d_ang = np.random.normal(0, 0.01)
            

        # before moving. try reorientation chance.
        rando = random.uniform(0, 1) * 100        
        if rando < reorientation_chance and new_v > 0:
            # reorientation make current direction equally randomly face anywhere.
            x, y, v, ang = states[-1]
            states[-1] = (x, y, v, np.random.uniform(- np.pi, np.pi)) 
                   
        # now generate all changes.
        x, y, v, ang = states[-1]
        for i in range(N_seg):
            x, y, v, ang = (x + v * np.cos(ang),
                            y + v * np.sin(ang),
                            new_v,
                            ang+d_ang)
            states.append((x, y, v, ang))        


    # remove first point (which was dummy values) and cut down to N_points points
    x, y, vel, thetas = zip(*states[2:N_points +2])
    return x, y, vel, thetas
'''
def noise_calc(raw_xyt, smooth_xyt):
    #"""
    Calculates a rough estimate of the noise by taking the smoothed data, pretending it is the 'real'
    path of the worm, and then calculating the standard deviation of the distribution of the displacement
    of the raw data from the smoothed data.
    :param raw_xyt: A list of xyt values from the raw worm data.
    :param smooth_xyt: A list of xyt values that have been smoothed such that it is supposed to represent the
    'real' worm path.
    :return: Returns separate x and y noise estimates.
    #"""
    dx = []
    dy = []
    for a, point in enumerate(raw_xyt):
        dp = np.array(point) - np.array(smooth_xyt[a])
        dx.append(dp[0])
        dy.append(dp[1])
    noise_x = np.std(dx)
    noise_y = np.std(dy)
    print noise_x, noise_y
    return noise_x, noise_y


def add_noise(raw_xyt, smooth_xyt):
    #"""
    Adds gaussian noise to xyt data that was smoothed from real worm raw xyt data. The size of the noise is estimated
    using the noise_calc function, and the values of the noise for each x and y value are randomly chosen from
    a random normal distribution.
    :param raw_xyt: A list of xyt values from the raw worm data.
    :param smooth_xyt: A list of xyt values that have been smoothed such that it is supposed to represent the
    'real' worm path.
    :return: Returns a list of xyt values that have had noise added to them.
    #"""
    noise_x, noise_y = noise_calc(raw_xyt, smooth_xyt)
    x, y, t = zip(*smooth_xyt)
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)
    n_x = x + np.random.normal(scale=noise_x, size=x.shape)
    n_y = y + np.random.normal(scale=noise_y, size=y.shape)
    noisy_xyt = zip(n_x.tolist(), n_y.tolist(), t)
    return noisy_xyt


def return_default_worm():
    #"""
    This function opens the json file, collects the xyt data, smooths and adds noise to the data, and
    returns a worm dictionary that contains the raw, smoothed, and noisy xy data, the time, and the
    pixels-per-mm of the worm. This is for the default worm that I have been testing.

    :return: A dictionary with keys 'raw-xy', 'smooth-xy', 'noisy-xy', 'pixels-per-mm', and 'time'.
    #"""
    worm_name = '20130319_161150_00006'
    raw_xyt, ppm = worm_xyt_extractor(worm_name)
    smooth_xyt = fu.smoothing_function(raw_xyt, window=11, order=5)
    noisy_xyt = add_noise(raw_xyt, smooth_xyt)
    worm = dict()
    rx, ry, rt = zip(*raw_xyt)
    sx, sy, st = zip(*smooth_xyt)
    nx, ny, nt = zip(*noisy_xyt)
    rxy = zip(rx, ry)
    sxy = zip(sx, sy)
    nxy = zip(nx, ny)
    worm['raw-xy'] = rxy
    worm['smooth-xy'] = sxy
    worm['noisy-xy'] = nxy
    worm['pixels-per-mm'] = ppm
    worm['time'] = rt
    return worm


def return_noisy_worm(worm_name):
    """
    Same as return_default_worm, except with a parameter for selecting the specific worm you want to use.

    This function opens the json file, collects the xyt data, smooths and adds noise to the data, and
    returns a worm dictionary that contains the raw, smoothed, and noisy xy data, the time, and the
    pixels-per-mm of the worm.
    :param worm_name: A string of the worm name - YYMMDD_HHMMSS_WORMID
    :return: A dictionary with keys 'raw-xy', 'smooth-xy', 'noisy-xy', 'pixels-per-mm', and 'time'.
    """
    raw_xyt, ppm = worm_xyt_extractor(worm_name)
    smooth_xyt = fu.smoothing_function(raw_xyt, window=11, order=5)
    noisy_xyt = add_noise(raw_xyt, smooth_xyt)
    worm = dict()
    rx, ry, rt = zip(*raw_xyt)
    sx, sy, st = zip(*smooth_xyt)
    nx, ny, nt = zip(*noisy_xyt)
    rxy = zip(rx, ry)
    sxy = zip(sx, sy)
    nxy = zip(nx, ny)
    worm['raw-xy'] = rxy
    worm['smooth-xy'] = sxy
    worm['noisy-xy'] = nxy
    worm['pixels-per-mm'] = ppm
    worm['time'] = rt
    return worm


def noisy_worm_from_plate(worm_name):
    #"""
    Same as return_default_worm, except with a parameter for selecting the specific worm you want to use.
    This function is to be used instead of the return_noisy_worm function when the worm that you want to use
    hasn't already had its data extracted from the raw plate json file. This function will extract the worm
    data from the plate data file first, then use the worm data as in the return_default_worm function.

    This function opens the json file, collects the xyt data, smooths and adds noise to the data, and
    returns a worm dictionary that contains the raw, smoothed, and noisy xy data, the time, and the
    pixels-per-mm of the worm.
    :param worm_name: A string of the worm name - YYMMDD_HHMMSS_WORMID
    #"""
    plate_name = worm_name[:-6]
    file_name = glob.glob('./../../Data/N2_agingxy_raw/{p}_*.json'.format(p=plate_name))
    plate_data = json.load(open(file_name[0], 'r'))
    worm_data = plate_data[worm_name]
    worm = dict()
    xy = worm_data['data']
    x, y = zip(*xy)
    t = worm_data['time']
    raw_xyt = zip(x, y, t)
    smooth_xyt = fu.smoothing_function(raw_xyt, window=11, order=5)
    noisy_xyt = add_noise(raw_xyt, smooth_xyt)
    worm['time'] = t
    worm['pixels-per-mm'] = worm_data['pixels-per-mm']
    sx, sy, st = zip(*smooth_xyt)
    nx, ny, nt = zip(*noisy_xyt)
    sxy = zip(sx, sy)
    nxy = zip(nx, ny)
    worm['smooth-xy'] = sxy
    worm['noisy-xy'] = nxy
    worm['raw-xy'] = xy
    print worm
'''

def add_set_noise(xyt, noise_x, noise_y=0):
    """
    Adds noise to a list of xyt points. The size of the noise is predetermined and brought into the function
    as a parameter.
    :param xyt: A list of xyt points, presumably the 'real' worm path.
    :param noise_x: A positive number representing the size of the normal distribution from which the noise
    will be randomly chosen for each point in the xyt list.
    :param noise_y: If no value is given, the y noise will be set equal to the x noise. Otherwise, this value
    is used the same way the noise_x is used, except for the y value of each xyt point.
    :return: A list of noisy xyt points representing the 'collected' noisy data
    """
    if noise_y == 0:
        noise_y = noise_x
    x, y, t = zip(*xyt)
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)
    n_x = x + np.random.normal(scale=noise_x, size=x.shape)
    n_y = y + np.random.normal(scale=noise_y, size=y.shape)
    noisy_xyt = zip(n_x.tolist(), n_y.tolist(), t)
    return noisy_xyt

def speeds_and_angles(times, x, y):
    x, y = list(x), list(y)
    dt = np.diff(np.array(times))    
    dx = np.diff(np.array(x))
    dy = np.diff(np.array(y))
    # to guard against division by zero
    for i, t in enumerate(dt):
        if t < 0.0000001:
            dt[i] = 0.0000001
    speeds = np.sqrt(dx**2 + dy**2) / dt
    speeds = list(speeds) + [np.nan]
    angles = [np.nan] + list(angle_change_for_xy(x, y, units='rad')) + [np.nan]
    return speeds, angles

def xy_to_full_dataframe(times, x, y):    
    speeds, angles = speeds_and_angles(times, x, y)
    print len(times), len(x), len(y), len(speeds), len(angles)
    df = pd.DataFrame(zip(x, y, speeds, angles), index=times,
                      columns=['x', 'y', 'v', 'dtheta'])
    print df.head()
    return df

def generate_bulk_tests(N=1, l=3600, noise_levels=[0.1, 0.2, 0.3], savedir='./'):
    noiseless = pd.DataFrame(index=range(l))
    soln = pd.DataFrame(index=range(l))
    for i in range(N):
        x, y, v, theta = generate_individual_series(N_points=l)
        noiseless['x{i}'.format(i=i)] = x
        noiseless['y{i}'.format(i=i)] = y
        soln[i] = [True if vi ==0 else False for vi in v]

    noiseless.to_hdf('{d}noiseless_xy.h5'.format(d=savedir), 'table')
    soln.to_hdf('{d}paused_soln.h5'.format(d=savedir), 'table')
    print noiseless.head()
    print soln.head()
    print 'noisless'
        
    noisy_sets = []
    for i, noise in enumerate(noise_levels):
        savename = 'noisy_xy_{n}.h5'.format(n=str(noise).replace('.', 'p'))
        print savename
        noise_array = np.random.normal(scale=noise, size=noiseless.shape)
        noisy = noiseless + noise_array
        print noisy.head()
        noisy.to_hdf('{d}{n}'.format(d=savedir, n=savename), 'table')

def create_synthetic_worm_csvs(noise_level=0.1):
    """
    """    
    x, y, v, theta = generate_individual_series()

    #t = np.arange(0, len(x)*0.1, 0.1)
    t = np.arange(0, len(x), 1)  
    soln = pd.DataFrame(index=t)
    
    dtheta = [np.nan] + [rescale_radians(r) for r in np.diff(theta)]
    soln['x'], soln['y'], soln['v'], soln['dtheta'] = x, y, v, dtheta
    soln.to_csv('soln.csv')    
    
    soln2 = xy_to_full_dataframe(t, x, y)
    soln2.to_csv('soln2.csv')    
            
    n_x, n_y, n_t = zip(*add_set_noise(zip(x, y, t), noise_level))
    noisy = xy_to_full_dataframe(n_t, n_x, n_y)       
    noisy.to_csv('noisy.csv')    

    s_x, s_y, s_t = zip(*fu.smoothing_function(zip(n_x, n_y, n_t), 225, 5))
    smoothed = xy_to_full_dataframe(s_t, s_x, s_y)       
    smoothed.to_csv('smoothed225.csv')    

    s_x, s_y, s_t = zip(*fu.smoothing_function(zip(n_x, n_y, n_t), 75, 5))
    smoothed = xy_to_full_dataframe(s_t, s_x, s_y)       
    smoothed.to_csv('smoothed75.csv')    

    s_x, s_y, s_t = zip(*fu.smoothing_function(zip(n_x, n_y, n_t), 35, 5))
    smoothed = xy_to_full_dataframe(s_t, s_x, s_y)       
    smoothed.to_csv('smoothed35.csv')    

    s_x, s_y, s_t = zip(*fu.smoothing_function(zip(n_x, n_y, n_t), 11, 5))
    smoothed = xy_to_full_dataframe(s_t, s_x, s_y)       
    smoothed.to_csv('smoothed11.csv')    
        
if __name__ == '__main__':
    #generate_bulk_tests(N=5, './data/smoothing/')
    create_synthetic_worm_csvs()
