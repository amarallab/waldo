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
from importing import angle_calculations as ac

def generate_individual_series():
    """
    Generates an entire individual timeseries
    """

    # toggles
    reorientation_chance = 10
    final_N = 36000
    

    state = (0, 0, 0, 0)
    states = [state]    

    while len(states) < final_N + 1:
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


    # remove first point (which was dummy values) and cut down to final_N points
    x, y, vel, thetas = zip(*states[1:final_N +1])
    return x, y, vel, thetas


def generate_individual_series2():
    """
    Generates an entire individual timeseries
    """
    # toggles
    reorientation_chance = 10
    final_N = 36000
    
    domains = []
    x, y, vel, thetas = [0.0], [0.0], [0.0], [0.0]
    #just_reoriented = False

    v = 0.0
    while len(x) < final_N:
        #How long this segment is going to be, a random int drawn from an exponential distribution.
        N_seg = int(sts.expon.rvs(scale=100))

        #Set the velocity
        #Velocity can currently be selected from three different options.
        #The three options are currently equally likely.
        v, last_v = random.choice([0, 0.01, 0.04]), v
        #All the instantaneous velocities of the segment are set to the same velocit.
        
        #if moving, angle change is chosen from a normal distribution.
        d_theta = 0
        if v > 0:
            d_theta = np.random.normal(0, 0.01)


        new_thetas = [thetas[-1] + (i * d_theta) for i in range(N_seg)]
        new_vel = v * np.ones(N_seg)        
        #if N_seg > 0 :
        new_x = [x[-1] + last_v * np.cos(thetas[-1])]
        new_y = [x[-1] + last_v * np.sin(thetas[-1])]

        for ang in new_thetas[1:]:
            dx = v * np.cos(ang)
            dy = v * np.sin(ang)
            new_x.append(new_x[-1] + dx)
            new_y.append(new_y[-1] + dy)
            
        #new_x = [x0 + (v * np.cos(ang)) for i, ang in enumerate(new_thetas)]
        #new_y = [y0 + (v * np.sin(ang)) for i, ang in enumerate(new_thetas)]
        check_list = [len(new_x), len(new_y), len(new_vel), len(new_thetas)]
        for i in check_list:
            if i != N_seg:
                print check_list, N_seg
                #print new_x
        if N_seg > 0:
            x.extend(new_x)
            y.extend(new_y)
            vel.extend(new_vel)
            thetas.extend(new_thetas)

        '''      
        for t in range(N_seg):
            #Change in x and y is determined by the direction of movement and the magnitude of the movement
            
            theta, last_theta = last_theta + d_theta, theta
            
            dx = vel[t] * np.cos(theta[-1])
            dy = vel[t] * np.sin(theta[-1])
            #Increment the position by the x and y change.
            x.extend(x[-1] + dx)
            y.extend(y[-1] + dy)
            #Slightly change the direction based on the angle change.
            theta.append(theta[-1] + float(d_theta))

        #Reorientation doesn't move the worm, just completely changes the direction.
        # any direction is equally probable.
        rando = random.uniform(0, 1) * 100        
        if rando < reorientation_chance and v>0:
            theta[-1] = np.random.uniform(0, 2 * np.pi)
            
            
        return x[1:], y[1:], vel[:-1], theta[1:], False


        
        xi, yi, veli, thetai, just_reoriented = generate_xy_pattern(x[-1], y[-1],
                                                                    vel[-1],
                                                                    theta[-1],
                                                                    just_reoriented)
        '''                                                                    
    print len(x), len(y), len(vel), len(thetas)        
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
    return x, y, vel, thetas, domains

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


def xy_to_full_dataframe(times, x, y):
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
    angles = [np.nan] + list(ac.angle_change_for_xy(x, y, units='rad')) + [np.nan]
    print len(times), len(x), len(y), len(speeds), len(angles)
    df = pd.DataFrame(zip(x, y, speeds, angles), index=times,
                      columns=['x', 'y', 'v', 'dtheta'])
    print df.head()
    return df
    
def create_synthetic_worm_csvs(noise_level=0.1):
    """
    """    
    x, y, v, theta = generate_individual_series()

    #t = np.arange(0, len(x)*0.1, 0.1)
    t = np.arange(0, len(x), 1)  
    soln = pd.DataFrame(index=t)
    '''
    def rescale_dthetas(dthetas):
        new_dth = []
        for dth in dthetas:
            steps = abs(int(dth / (2*np.pi))) + 1
            if dth < -np.pi:
                dth += steps * 2 * np.pi
            elif dth > np.pi:
                dth -= steps * 2 * np.pi
            new_dth.append(dth)
        return new_dth
    '''
    
    dtheta = [np.nan] + [ac.rescale_radians(r) for r in np.diff(theta)]
    soln['x'], soln['y'], soln['v'], soln['dtheta'] = x, y, v, dtheta
    soln.to_csv('soln.csv')    
    
    soln2 = xy_to_full_dataframe(t, x, y)
    soln2.to_csv('soln2.csv')    
            
    n_x, n_y, n_t = zip(*add_set_noise(zip(x, y, t), noise_level))
    noisy = xy_to_full_dataframe(n_t, n_x, n_y)       
    noisy.to_csv('noisy.csv')    

    s_x, s_y, s_t = zip(*fu.smoothing_function(zip(n_x, n_y, n_t), 11, 5))
    smoothed = xy_to_full_dataframe(s_t, s_x, s_y)       
    smoothed.to_csv('smoothed.csv')    
        

if __name__ == '__main__':
    #Call a function that will create a worm dictionary, like synthetic_worm_creator(), noisy_worm_from_plate,
    #return_noisy_worm, or return_default_worm
    #worm = synthetic_worm_creator()
    #This worm is then saved to a location of your choosing.
    #save_dir = '{d}/smoothing'.format(d=TEST_DATA_DIR)
    #json.dump(worm, open('{d}/synthetic_1.json'.format(d=save_dir), 'w'), indent=4)
    #print 'Hello'
    create_synthetic_worm_csvs()
