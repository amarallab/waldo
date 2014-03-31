"""
This serves as a holding file for functions used to take the raw data for the position points of the worm
smooth those points using a savitzky-golay polynomial smoothing. This is done over a range of window and order
sizes. Those smoothed points are then used to calculate values of possible experimental interest, which are
saved as json files, along with the window size and order used to smooth them.
"""
import os
import sys
import json
import glob
import numpy as np

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = HERE + '/../'
sys.path.append(SHARED_DIR)

from filtering.filter_utilities import smoothing_function

def calc_total_length(xyt):
    """
    This function takes as a parameter a list of points. It passes this list to a function that calculates
    the vectors, magnitudes, directions, and times (averaged between the start and end point of each vector).
    This particular function then takes the magnitudes and sums them to obtain the total distance travelled by
    the worm.
    """
    vectors, magnitudes, thetas, vec_times = calc_vector(xyt)
    return sum(magnitudes)


def calc_speeds(xyt):
    vectors, magnitudes, thetas, vec_times = calc_vector(xyt)
    time_diff = []
    for i, p in enumerate(xyt[: -1]):
        time_diff.append(xyt[i+1][2] - p[2])
    speeds = [x/y for x, y in zip(magnitudes, time_diff)]
    return speeds


def calc_median_speed(xyt):
    """
    This function takes in a list of x, y, and time values of the worm position, uses the calc_vector
    function to calculate the magnitudes and vector times (averages between the time values of the start
    and end points of the vector). The difference in times between the start and end point of the vectors
    are then used, and the magnitudes are divided by the difference in time to get the speed. The median
    is then taken of this speed data, and returned.
    """
    speeds = calc_speeds(xyt)
    return np.median(speeds)


def calc_mean_speed(xyt):
    """
    This function takes in a list of x, y, and time values of the worm position, uses the calc_vector
    function to calculate the magnitudes and vector times (averages between the time values of the start
    and end points of the vector). The difference in times between the start and end point of the vectors
    are then used, and the magnitudes are divided by the difference in time to get the speed. The mean
    is then taken of this speed data, and returned.
    """
    speeds = calc_speeds(xyt)
    return np.mean(speeds)


def calc_delta_thetas(xyt):
    vectors, magnitudes, thetas, vec_times = calc_vector(xyt)
    dot_theta, theta_times = cross_product(vectors, vec_times)
    return dot_theta


def calc_mean_theta(xyt):
    """
    Takes in the list of positions, uses calc_vector to turn the positions into vectors. the vectors are then
    passed to angle_between, which returns the angles between the vectors (change in direction). The mean change
    in direction is then calculated and returned.
    """
    dot_theta = calc_delta_thetas(xyt)
    return np.mean(dot_theta)


def calc_median_theta(xyt):
    """
    Takes in the list of positions, uses calc_vector to turn the positions into vectors. the vectors are then
    passed to angle_between, which returns the angles between the vectors (change in direction).The median
    change in direction is then calculated and returned.
    """
    dot_theta = calc_delta_thetas(xyt)
    return np.median(dot_theta)


def calc_theta_over_threshold(xyt, time_span=5, thresh=90):
    """
    This function takes in the position list, along with a set time-span and a threshold of movement. The time-span
    represents the amount of time the worm will move before an angle is measured -- this allows for the generation
    of directional change that covers a longer time span than the individual position measurements, which are done
    at around the rate of 1/10th of a second. The threshold value is the size of the directional change that must
    be overcome before the "direction change" counter is incremented.
    """
    vectors, magnitudes, thetas, vec_times = calc_vector(xyt)
    dot_theta, theta_times = cross_product(vectors, vec_times)
    counter = 0
    start = 0
    s = 0
    for i, time in enumerate(theta_times[: -1]):
        s += theta_times[i+1]-time
        if s >= time_span:
            s = 0
            angle = sum(dot_theta[start:i])
            start = i
            if angle >= thresh:
                counter += 1
    return counter


def calc_angular_travelled(xyt):
    """Takes in the list of positions, uses calc_vector to turn the positions into vectors. the vectors are then
    passed to angle_between, which returns the angles between the vectors (change in direction). This is then
    summed and returned, representing the total angular distance travelled, in degrees.
    """
    vectors, magnitudes, thetas, vec_times = calc_vector(xyt)
    d_theta, d_time = cross_product(vectors, vec_times)
    return sum(d_theta)


def angle_between(v_array, vector_times):
    """
    Takes in an array of vectors and returns an array of the angles between the vectors. This function does
    so by using the dot product, and hence uses the arccosine function. Arccosine can be sensitive to values
    near -1 and 1, and is undefined for values outside of (-1, 1). Due to rounding issues, values greater than
    1 and less than -1 sometimes arise -- this is corrected in a pretty sloppy fashion by setting those values
    to very near 1 and -1, at arbitrary accuracy. There might be a better way to deal with this issue.

    The function returns the angles between the vectors, along with a time value that matches the size of the
    angle list (which is the size of the vector list - 1). This time list was done by averaging every two values
    of the vector times.
    :param v_array: A list of 3 value lists, with the values corresponding to the i and j values of the
    position vector of the worm, and the third value corresponding to the duration of that vector.
    :param vector_times: A list of values representing the time during the filming of the worm movement,
    generally not beginning at zero. These values correspond by index with the vectors in the v_array parameter.
    """
    dot_theta = []
    delta_time = []
    #Loop through the vectors to find the angle between each consecutive pair of vectors
    for a, vector in enumerate(v_array[: -1]):
        #mA and mB are the magnitudes of the a and a+1 vector, respectively
        #Since the vectors within the v_array have a third value, a time component,
        #it is important to find the magnitude using only the x and y components.
        mA = np.linalg.norm(np.array([v_array[a][0], v_array[a][1]]))
        mB = np.linalg.norm(np.array([v_array[a+1][0], v_array[a+1][1]]))
        #The pre-angle represents the value of cosine(theta), where theta is the angle between
        #vectors a and a+1. This value must necessarily be less than 1, but due to float rounding
        #issues in python, the pre-angle must be checked and bounded to be within 1.
        pre_angle = (v_array[a][0]*v_array[a+1][0] + v_array[a][1]*v_array[a+1][1]) / mA / mB
        if pre_angle >= 1:
            pre_angle = 0.999999999
        if pre_angle <= -1:
            pre_angle = -0.999999999
        dot_theta.append(np.arccos(pre_angle) * 180 / np.pi)
        #Delta time represents the average of the time points of the two vectors, or the time during
        #the filming that the theta occurred.
        delta_time.append((vector_times[a+1]+vector_times[a]) / 2)
    return dot_theta, delta_time


def cross_product(v_array, v_times):
    thetas = []
    dot_thetas, theta_times = angle_between(v_array, v_times)
    for a, vector in enumerate(v_array[:-1]):
        v1 = np.array([vector[0], vector[1]])
        v2 = np.array([v_array[a+1][0], v_array[a+1][1]])
        mv3 = np.cross(v1, v2)
        theta = np.arcsin(mv3 / np.linalg.norm(v1) / np.linalg.norm(v2))
        if dot_thetas[a] > 90 and theta > 0:
            thetas.append(180 - (theta * 180 / np.pi))
        elif dot_thetas[a] > 90 and theta < 0:
            thetas.append(-180 - (theta * 180 / np.pi))
        else:
            thetas.append(theta * 180 / np.pi)
    return thetas, theta_times


def calc_vector(points):
    """
    :param points: A list of a 3-value list of x, y, and t values.

    Takes in a list of x, y, and t points, and returns: a list of vectors in the format of i, j, and dt,
    where dt represents the duration of the vector; a list of the magnitudes of the vectors; a list of
    the directions of the vectors, with the positive x-axis representing zero, and the negative x-axis
    representing 180 or -180, depending on the y-axis sign; and a list of vec-times, which represent the
    time during filming that a given vector took place.
    """
    vectors = []
    magnitudes = []
    thetas = []
    vec_times = []
    #Iterate through the xyt points and calculate the values needed for the four outputs
    for a, xyt in enumerate(points[: -1]):
        #Subtract the 'a' xyt from the 'a+1' xyt to get the vector
        vectors.append(np.subtract(points[a+1], xyt))
        #Find the magnitude of the i and j values of the vector (not the dt!)
        magnitudes.append(np.linalg.norm(np.array([vectors[a][0], vectors[a][1]])))
        #Use the j value of the vector to determine if the direction of the vector should
        #be positive or negative
        if vectors[a][1] >= 0:
            thetas.append(np.arccos(vectors[a][0] / magnitudes[a]) * 180 / np.pi)
        else:
            thetas.append(-np.arccos(vectors[a][0] / magnitudes[a]) * 180 / np.pi)
        #Average the time values of the 'a' and 'a+1' xyt to find the vector time
        vec_times.append((xyt[2] + points[a+1][2]) / 2)
    return vectors, magnitudes, thetas, vec_times




def angular_travel(angle_array):
    """
    Takes in an array of angular change and returns the angles travelled. This is sensitive to changes in
    direction, The angles travelled are returned as an array of stepwise values
    """
    travelled = [0]
    for a, angle in enumerate(angle_array):
        travelled.append(travelled[a] + angle)

    return travelled


def xyt_theta_threshold(xyt, time_span=5, upper=90, lower=0):
    """
    Right now, this function takes in the xyt positions as a list, the time span that you want to cover in
    each calculation for turning (this may be changed in the future to be a set distance instead of a set
    time), and the threshold for a directional change.

    Right now, the function returns a list of points that are considered the middle of a change of direction,
    calculated over a set period of time.
    """
    vectors, magnitudes, thetas, vec_times = calc_vector(xyt)
    dot_theta, theta_times = cross_product(vectors, vec_times)
    start = 0
    s = 0
    threshold_xyt = []
    for i, time in enumerate(theta_times[: -1]):
        s += theta_times[i+1]-time
        if s >= time_span:
            s = 0
            angle = sum(dot_theta[start:i])
            if lower <= angle >= upper:
                middle_index = (i + start) / 2
                midpt = xyt[middle_index]
                threshold_xyt.append([midpt[0], midpt[1], midpt[2]])
            start = i
    data = {'Threshold_Points': threshold_xyt, 'Smoothed_Points': xyt}
    return data


def loop_through_things(func, prop, search_dir, save_dir, window_sizes=range(11, 301, 2), orders=range(1, 6, 2)):
    """
    Takes in the parameters for running the smoothing function, running a calculation, and saving the calculated data
    to a .json file. the property parameter is used to name the json file. The function first loads the raw data from
    a .json file, and stores the x, y, and time values into lists. It then passes this list into a smoothing function
    before passing the output into whatever calculation function was specified in the func parameter. This is
    iterated over the range of window sizes for each order in the range of orders
    """
    #range(11, 101, 2) + range(101, 1001, 10)
    # read the data file
    #Loading in the json files
    json_data_files = glob.glob('{d}/*.json'.format(d=search_dir)
    print json_data_files
    for json_filename in json_data_files:
        worm_ID = json_filename[json_filename.find('Raw') +
                                4:json_filename.find('_', json_filename.find('_', json_filename.find('_')+1)+1)]
        print json_filename
        file_object = open(json_filename, 'r')
        data = json.load(file_object)
        print data.keys()
        times = data['time']
        raw_xy = data['data']
        raw_x, raw_y = zip(*raw_xy)
        raw_xyt = zip(raw_x, raw_y, times)

        results_dict = dict()
        results_dict['data'] = []
        results_dict['window_sizes'] = []
        results_dict['orders'] = []
        for order in orders:
            for window in window_sizes:
                results_dict['data'].append(func(smoothing_function(raw_xyt, window, order)))
                results_dict['window_sizes'].append(window)
                results_dict['orders'].append(order)
        #results = [[[func(smoothing_function(raw_xyt, win, order)), win] for win in window_sizes] for order in orders]
        #results_dict = {'data': results, 'order': orders}
        json.dump(results_dict, open('{d}/{p}_{w}.json'.format(d=save_dir, p=prop, w=worm_ID), 'w'),
                  indent=4)
    return []


if __name__ == '__main__':
    loop_through_things(func=calc_speeds, prop='speeds', save_dir='PropertyDistributions',
                        window_sizes=[11, 51, 71, 111, 201], orders=[5])
    print "Hello"
