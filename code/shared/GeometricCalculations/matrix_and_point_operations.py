
__author__ = 'peterwinter'

import os
import sys
#import pylab as pl
#import matplotlib.pyplot as pl

# path definitions
SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(SHARED_DIR)

# nonstandard imports
from ExceptionHandling.record_exceptions import write_pathological_input

'''
def plot_np_arrays(matrix_list):
    pl.figure()

    for matrix in matrix_list:
        points = []
        for i, matrix_row in enumerate(matrix):
            for j, b in enumerate(matrix_row):
                if matrix[i][j] == 1:
                    points.append((i, j))
        xs, ys = zip(*points)
        pl.plot(xs, ys, 'o') #marker='.')
    pl.show()


def plot_np_array(matrix, pt_list=[]):
    pl.figure()

    points = []
    for i, matrix_row in enumerate(matrix):
        for j, _ in enumerate(matrix_row):
            if matrix[i][j] == 1:
                points.append((i, j))
    if len(points) > 0:
        xs, ys = zip(*points)
        pl.plot(xs, ys, 'o')
    if len(pt_list) > 0:
        x, y = zip(*pt_list)
        pl.plot(x, y, marker='x', color='red', alpha=0.5)
    pl.show()
'''

def matrix_to_unordered_points(matrix):
    sm = matrix.copy()
    points = []
    for i, sm_row in enumerate(sm):
        for j, _ in enumerate(sm_row):
            if sm[i][j] == 1:
                points.append((i,j))
    return points


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
        counter = 0
        for i in range(1, 8):
            if P[i] == 0 and P[i + 1] == 1:
                counter += 1
        if P[8] == 0 and P[1] == 1:
            counter += 1
        return counter

    # since we know no points are at edge of matrix (previous assertions) loop through all non-edge points.
    for i, im_row in enumerate(im[1:-1], start=1):
        for j, _ in enumerate(im_row[1:-1], start=1):
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
