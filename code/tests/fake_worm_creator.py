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

TEST_DIR = os.path.dirname(os.path.realpath(__file__)) 
PROJECT_DIR = os.path.abspath(TEST_DIR + '/../../')
SHARED_DIR = PROJECT_DIR + '/code/shared/'
TEST_DATA_DIR = TEST_DIR + '/data/'

sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR + '/code/')

# nonstandard imports
import filtering.filter_utilities as fu

def generate_individual_series():
    """
    Generates an entire individual timeseries
    """
    domains = []
    x, y, vel, theta = [0.0], [0.0], [0.0], [0.0]
    while len(x) < 36000:
        xi, yi, veli, thetai = generate_xy_pattern(x[-1], y[-1], vel[-1], theta[-1])
        x.extend(xi)
        y.extend(yi)
        vel.extend(veli)
        theta.extend(thetai)
    in_domain = False
    domain_start = 0
    for a, v in enumerate(vel):
        if in_domain and v != 0:
            in_domain = False
            domain_end = a-1
            domains.append([domain_start, domain_end])
        elif not in_domain and v == 0:
            in_domain = True
            domain_start = a
    return x, y, vel, theta, domains

def show_series(x, y, vel, theta, domains):
    plt.figure(1)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.xlabel('y')

    plt.figure(2)
    ax = plt.subplot(411)
    plt.plot(x)
    plt.ylabel('x')
    plt.subplot(412, sharex=ax)
    plt.plot(y)
    plt.ylabel('y')
    plt.subplot(413, sharex=ax)
    plt.plot(vel)
    plt.ylabel('vel')
    plt.subplot(414, sharex=ax)
    plt.plot(theta)
    plt.ylabel('theta')
    plt.show()

def generate_xy_pattern(x0, y0, v0, theta0):
    rando = random.uniform(0, 1) * 100
    reorientation_chance = 10
    #Reorientation doesn't move the worm, just completely changes the direction.
    if rando < reorientation_chance:
        d_theta = np.random.normal(0, np.pi)
        return [x0], [y0], [v0], [d_theta]

    #How long this segment is going to be, a random int drawn from an exponential distribution.
    N = int(sts.expon.rvs(scale=100))
    times = range(N)

    #Set initial conditions
    x = [x0]
    y = [y0]
    theta = [theta0]
    
    #Set the velocity
    #Velocity can currently be selected from three different options.
    vs = [0, 0.01, 0.04]
    #The three options are currently equally likely.
    r_v = random.randint(0, 2)
    v = vs[r_v]
    #All the instantaneous velocities of the segment are set to the same velocity.
    vel = v * np.ones(N)
    #Angle change is chosen from a normal distribution.
    d_theta = np.random.normal(0, 0.01)

    for t in times[1:]:
        #Change in x and y is determined by the direction of movement and the magnitude of the movement
        dx = vel[t] * np.cos(theta[-1])
        dy = vel[t] * np.sin(theta[-1])
        #Increment the position by the x and y change.
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        #Slightly change the direction based on the angle change.
        theta.append(theta[-1] + float(d_theta))
    return x[1:], y[1:], vel[1:], theta[1:]


def worm_xyt_extractor(worm_name):
    """
    A function that returns the data for a specific worm that has already had its data saved into an
    individual json file. It extracts the x, y, and t data and zips them together, and also extracts
    the pixels-per-mm and returns that as well.

    :param worm_name: A string containing the YYMMDD_HHMMSS_WORMID name of the worm.
    """
    file_name = glob.glob('./../../Data/Raw/{w}_*.json'.format(w=worm_name))
    file_name = file_name[0]
    data = json.load(open(file_name, 'r'))
    xy = data['data']
    t = data['time']
    ppm = data['pixels-per-mm']
    x, y = zip(*xy)
    xyt = zip(x, y, t)
    return xyt, ppm


def noise_calc(raw_xyt, smooth_xyt):
    """
    Calculates a rough estimate of the noise by taking the smoothed data, pretending it is the 'real'
    path of the worm, and then calculating the standard deviation of the distribution of the displacement
    of the raw data from the smoothed data.
    :param raw_xyt: A list of xyt values from the raw worm data.
    :param smooth_xyt: A list of xyt values that have been smoothed such that it is supposed to represent the
    'real' worm path.
    :return: Returns separate x and y noise estimates.
    """
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
    """
    Adds gaussian noise to xyt data that was smoothed from real worm raw xyt data. The size of the noise is estimated
    using the noise_calc function, and the values of the noise for each x and y value are randomly chosen from
    a random normal distribution.
    :param raw_xyt: A list of xyt values from the raw worm data.
    :param smooth_xyt: A list of xyt values that have been smoothed such that it is supposed to represent the
    'real' worm path.
    :return: Returns a list of xyt values that have had noise added to them.
    """
    noise_x, noise_y = noise_calc(raw_xyt, smooth_xyt)
    x, y, t = zip(*smooth_xyt)
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)
    n_x = x + np.random.normal(scale=noise_x, size=x.shape)
    n_y = y + np.random.normal(scale=noise_y, size=y.shape)
    noisy_xyt = zip(n_x.tolist(), n_y.tolist(), t)
    return noisy_xyt


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


def return_default_worm():
    """
    This function opens the json file, collects the xyt data, smooths and adds noise to the data, and
    returns a worm dictionary that contains the raw, smoothed, and noisy xy data, the time, and the
    pixels-per-mm of the worm. This is for the default worm that I have been testing.

    :return: A dictionary with keys 'raw-xy', 'smooth-xy', 'noisy-xy', 'pixels-per-mm', and 'time'.
    """
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
    """
    Same as return_default_worm, except with a parameter for selecting the specific worm you want to use.
    This function is to be used instead of the return_noisy_worm function when the worm that you want to use
    hasn't already had its data extracted from the raw plate json file. This function will extract the worm
    data from the plate data file first, then use the worm data as in the return_default_worm function.

    This function opens the json file, collects the xyt data, smooths and adds noise to the data, and
    returns a worm dictionary that contains the raw, smoothed, and noisy xy data, the time, and the
    pixels-per-mm of the worm.
    :param worm_name: A string of the worm name - YYMMDD_HHMMSS_WORMID
    """
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


def synthetic_worm_creator(noise_level=0.1):
    """
    Uses the generate_individual_series function from generate_random_series.py to create a synthetic
    'true' worm path. This 'true' worm path then has noise added to it of a set value. The worm is then
    packaged into a dictionary, in as similar a way as the worms in return_default_worm and return_noisy_worm,
    which means that, in this case, the 'raw-xy' key is redundant. Of note, the 'pixels-per-mm' value is set
    to a good estimate of 40, and this dictionary contains an additional key, 'domains', which contains the
    true domains of non-movement of the worm.

    :return: A dictionary with keys 'raw-xy', 'smooth-xy', 'noisy-xy', 'pixels-per-mm', 'domains' and 'time'.
    """
    worm = dict()
    x, y, v, theta, domains = generate_individual_series()
    t = np.arange(0, len(x)*0.1, 0.1)
    raw_xyt = zip(x, y, t)
    raw_xy = zip(x, y)
    noisy_xyt = add_set_noise(raw_xyt, noise_level)
    n_x, n_y, n_t = zip(*noisy_xyt)
    n_xy = zip(n_x, n_y)
    worm['smooth-xy'] = raw_xy
    worm['noisy-xy'] = n_xy
    worm['raw-xy'] = raw_xy
    worm['time'] = t.tolist()
    worm['pixels-per-mm'] = 40
    worm['domains'] = domains
    return worm

if __name__ == '__main__':
    #Call a function that will create a worm dictionary, like synthetic_worm_creator(), noisy_worm_from_plate,
    #return_noisy_worm, or return_default_worm
    worm = synthetic_worm_creator()
    #This worm is then saved to a location of your choosing.
    save_dir = '{d}/smoothing'.format(d=TEST_DATA_DIR)
    json.dump(worm, open('{d}/synthetic_1.json'.format(d=save_dir), 'w'), indent=4)
    #print 'Hello'
