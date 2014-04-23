#!/usr/bin/env python

'''
Filename: skeletonize_outline.py
Description: 
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
import time

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
shared_directory = project_directory + 'code/shared/'
exception_directory = project_directory + 'data/importing/exceptions/'

assert os.path.exists(project_directory), 'project directory not found'
assert os.path.exists(shared_directory), 'shared code directory not found'
sys.path.append(shared_directory)

# if directory to store exceptions does not exist, create it.
if not os.path.exists(exception_directory):
    os.makedirs(exception_directory)
    
# nonstandard imports
from ExceptionHandling.record_exceptions import write_pathological_input
from GeometricCalculations.shape_thinning import skeletonize
#from GeometricCalculations.matrix_and_point_operations import compute_closest_point
#from GeometricCalculations.matrix_and_point_operations import close_outline_border
#from GeometricCalculations.matrix_and_point_operations import line_matrix_to_ordered_points
#from GeometricCalculations.matrix_and_point_operations import calculate_branch_and_endpoints

def trim_spine_matrix(spine_matrix, endpoints, branchpoints):
    '''
    removes one branch from a spine

    :param spine_matrix: binary numpy matrix containing 1s where the spine is and 0s all around.
    :param endpoints: list of tuples containing the x,y coordinates of the ends of a spine
    :param branchpoints: list of tuples containing x,y coordinates of each branchpoint along the spine
    '''
    sm = spine_matrix.copy()

    def cut_arm(spine_matrix, point_list):
        for pt in point_list: spine_matrix[pt[0]][pt[1]] = 0
        return spine_matrix

    # initialize lists from each endpoint, starting with that endpoint
    paths_from_endpoints = {}
    for k, ep in enumerate(endpoints):
        if k not in paths_from_endpoints:
            paths_from_endpoints[k] = []
            paths_from_endpoints[k].append(ep)
            sm[ep[0]][ep[1]] = 0

    # move each endpoint list progressivly towards a branchpoint
    while True:
        #for each branch compute next step
        for k in paths_from_endpoints:
            new_pt = compute_closest_point(sm, paths_from_endpoints[k][-1])
            if new_pt == None:
                write_pathological_input((spine_matrix.tolist(), endpoints),
                                         input_type='spine_matrix/endpoints', note='next point error',
                                         savename='%strim_spine_%s.json' %(exception_directory, str(time.time())))
            # if newpoint is a branchpoint, stop
            # if newpoint returns None, that is a signal to cut arm.
            if new_pt in branchpoints or new_pt == None:
                sm = cut_arm(spine_matrix, paths_from_endpoints[k])
                endpoints, branchpoints = calculate_branch_and_endpoints(sm)
                return sm, endpoints, branchpoints

            # if not, delete newpoint from image and add it to endpoint path
            else:
                sm[new_pt[0]][new_pt[1]] = 0
                paths_from_endpoints[k].append(new_pt)

def cut_branchpoints_from_spine_matrix(spine_matrix):
    '''
    repeatedly calls trim_spine_matrix in order to remove all branches in a particular spine matrix.
    this leaves a matrix containing a snakeing line of one pixel width.

    :param spine_matrix: binary numpy matrix containing 1s where the spine is and 0s all around.
    '''
    endpoints, branchpoints = calculate_branch_and_endpoints(spine_matrix)
    # having more than two endpoints means there are some branchpoints. this removes them.
    while len(endpoints) > 2:
        spine_matrix, endpoints, branchpoints = trim_spine_matrix(spine_matrix, endpoints, branchpoints)
    
    if len(endpoints) < 2: return spine_matrix, endpoints
    assert len(endpoints) == 2, ('there should be two endpoints, not: ' + str(endpoints)
                                 + '\n' + str(spine_matrix))
    return spine_matrix, endpoints


def compute_skeleton_from_outline(outline, return_intermediate_steps=False, verbose=False):
    '''
    accepts a list of points in the outline of a shape and returns the winding centerline 1 pixel wide of that shape

    :param outline: a list of x,y tuples containing the points around the shape's outline
    :param return_intermediate_steps: a toggle to show internal guts of the process.
    '''
    # occasional outlines do not form a fully closed shape. this closes it.
    xs, ys = zip(*close_outline_border(outline))
    # make a box that is big enough to hold the outline + borders + a safety margin
    border_size, safety_margin = 1, 1
    matrix_x = max(xs) - min(xs) + 2 * border_size + safety_margin
    matrix_y = max(ys) - min(ys) + 2 * border_size + safety_margin
    outline_matrix = np.zeros([matrix_x, matrix_y], dtype=int)

    # add outline to the matrix by shifting the xy coordinates accordingly
    xy_shift = [min(xs) - border_size, min(ys) - border_size]
    for x, y in zip(xs, ys):
        outline_matrix[x - xy_shift[0]][y - xy_shift[1]] = 1

    # fill in the inside of the outline to make a solid shape
    filled_matrix = binary_fill_holes(outline_matrix)
    # if something is wrong, save the offending outline for later
    if sum(sum(outline_matrix)) == sum(sum(filled_matrix)):
        write_pathological_input(outline.tolist(), input_type='outline', note='outline filling problem',
                                 savename='%sfill_outline_%s.json' %(exception_directory, str(time.time())))
        assert False, 'you have an outline filling problem'

    # thin the solid shape until it is 1px thick, then remove all shortest branches.
    spine_matrix_branched = skeletonize(filled_matrix)
    if return_intermediate_steps:
        spine_matrix_branched_copy = spine_matrix_branched.copy()
    spine_matrix, endpoints = cut_branchpoints_from_spine_matrix(spine_matrix_branched)

    # the option to return matrices from intermediate steps is for plotting
    if return_intermediate_steps:
        return outline_matrix, filled_matrix, spine_matrix_branched_copy, spine_matrix, xy_shift, endpoints

    if len(endpoints) < 2:
        if sum(sum(spine_matrix)) > 2:
            # if spine is long enough to have endpoints, something is wrong.
            # this saves the offending outline for later tests
            write_pathological_input(outline.tolist(),
                                     input_type='outline', note='less than 2 endpoints. long enough spine.',
                                     savename='%sendpoints_%s.json' %(exception_directory, str(time.time())))
            assert False, 'spine is long enough, but does not have endpoints' + str(endpoints)
        else:
            # short spines can't be helped.
            if verbose:
                print 'warning: spine too short to find endpoints!'
            return []

    # change spine matrix back to a list of points and reverse the previous coordinate shift
    shifted_spine = line_matrix_to_ordered_points(spine_matrix, endpoints)
    real_spine = [(pt[0] + xy_shift[0], pt[1] + xy_shift[1]) for pt in shifted_spine]
    return real_spine


def compute_closest_point(im, pt, safe_mode=False):
    '''
    given a matrix 'im', find the single closest point to point 'pt'. returns one point or None.

    inputs
    im -- a binary matrix
    pt -- an xy tuple specifying the curent location
    safe_mode -- boolean toggle. true = assert statements stop code

    output
    (x, y) -- coordinates of nearest point
    '''
    i, j = pt
    nearest_neighbors = [(i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j)]
    diagonal_neighbors = [(i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]
    actual_neighbors = []
    #print 'im size', len(im), len(im[0])
    #print nearest_neighbors
    # first we check neighbors which are closer (not on the diagonal)
    for n in nearest_neighbors:
        # make sure none of neighbors are out of bounds.
        if n[0] < len(im) and n[1] < len(im[0]):
            if im[n[0]][n[1]] == 1:
                actual_neighbors.append(n)

    # if no points are aligned, check diagonals
    if len(actual_neighbors) == 0:
        for n in diagonal_neighbors:
            if n[0] < len(im) and n[1] < len(im[0]):
                if im[n[0]][n[1]] == 1:
                    actual_neighbors.append(n)

    # this code should only return one nearest neighbor
    if safe_mode:
        assert len(actual_neighbors) < 2, (str(pt) + ' has too many closest neighbors' +
                                           str(actual_neighbors) + '\n' + str(im))
        assert len(actual_neighbors) > 0, (str(pt) + ' has not enough closest neighbors' +
                                           str(actual_neighbors) + '\n' + str(im))
    else:
        if len(actual_neighbors) > 1: return None
        if len(actual_neighbors) == 0: return None

    return actual_neighbors[0]


def close_outline_border(outline, verbose=False):
    '''
    '''
    pt_1, pt_N = outline[0], outline[-1]
    need_filler = False
    new_x, new_y = pt_N

    # if there is more than one pixel difference
    # make new last point one closer than old last point
    if (pt_1[0] - pt_N[0]) < -1:
        need_filler = True
        new_x -= 1
    elif (pt_1[0] - pt_N[0]) > 1:
        need_filler = True
        new_x += 1
    elif (pt_1[1] - pt_N[1]) < -1:
        need_filler = True
        new_y -= 1
    elif (pt_1[1] - pt_N[1]) > 1:
        need_filler = True
        new_y += 1

    if need_filler:
        if verbose:
            print 'fixing:', pt_1, (new_x, new_y), pt_N
        outline.append((new_x, new_y))
        # if one gap needed to be filled, check recursively to see if a gap still present.
        close_outline_border(outline)
    return outline


def line_matrix_to_ordered_points(matrix, endpoints):
    sm = matrix.copy()
    assert len(endpoints) == 2, ('there should be two endpoints, not: ' + str(endpoints))
    firstpoint = endpoints[0]
    points = [firstpoint]
    sm[firstpoint[0], firstpoint[1]] = 0
    while True:
        next_pt = compute_closest_point(im=sm, pt=points[-1])
        if next_pt == None:
            write_pathological_input((matrix.tolist(), endpoints), input_type='spine_matrix/endpoints', note='next point error')
            assert next_pt != None, 'next point error'

        #print next_pt, points[-1]
        points.append(next_pt)
        #print points
        sm[next_pt[0], next_pt[1]] = 0
        #plot_np_array(sm, points)
        #print 'endpoints', endpoints
        if next_pt == endpoints[1]:
            #print 'hit endpoint'
            break
    return points


def calculate_branch_and_endpoints(spine_matrix):
    im = spine_matrix
    endpoints = []
    branchpoints = []

    # make sure none of your points are at the edge of the matrix. if messes up the algorithm.
    er_msg = 'error cant have points on edge of matrix'
    # check top and bottom rows.
    assert sum(im[0]) == 0, er_msg
    assert sum(im[-1]) == 0, er_msg
    # check first and last columns.
    for j, _ in enumerate(im):
        assert im[j][0] == 0, er_msg
        assert im[j][-1] == 0, er_msg

    def count_neighbor_chains(P):
        ''' circle point and count times pixels go from '0' to '1'.
        this gives
        '''
        #counter = 0
        #for i in range(1, 8):
        #    if P[i] == 0 and P[i + 1] == 1:
        #        counter += 1
        #if P[8] == 0 and P[1] == 1:
        #    counter += 1
        counter = 0
        prev = P[8]
        for cur in P[1:9]:
            if prev == 0 and cur == 1:
                counter += 1
            prev = cur
        return counter

    # since we know no points are at edge of matrix (previous assertions) loop through all non-edge points.
    #for i, im_row in enumerate(im[1:-1], start=1):
    #    for j, _ in enumerate(im_row[1:-1], start=1):
    DIM_Y = len(im)
    DIM_X = len(im[0])
    for i in range(1, DIM_Y - 1):
        for j in range(1, DIM_X - 1):
            # if the point is filled in, check neighbors
            if im[i][j] > 0:
                P = [im[i][j],
                     im[i - 1][j],
                     im[i - 1][j + 1],
                     im[i][j + 1],
                     im[i + 1][j + 1],
                     im[i + 1][j],
                     im[i + 1][j - 1],
                     im[i][j - 1],
                     im[i - 1][j - 1]]
                # if point has only one neighbor, it is an end point
                if count_neighbor_chains(P) == 1:
                    endpoints.append((i, j))
                # if point has 3 or more neighbors, it is a branch point
                if count_neighbor_chains(P) >= 3:
                    branchpoints.append((i, j))

    return endpoints, branchpoints


def matrix_to_unordered_points(matrix):
    sm = matrix.copy()
    points = []
    for i, sm_row in enumerate(sm):
        for j, _ in enumerate(sm_row):
            if sm[i][j] == 1:
                points.append((i,j))
    return points