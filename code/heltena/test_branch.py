import profiling

EXAMPLE_SPLINE = [
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000] ]

def peter_calculate_branch_and_endpoints(spine_matrix):
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


def heltena_calculate_branch_and_endpoints(spine_matrix):
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
        prev = P[8]
        for cur in P[1:9]:
            if prev == 0 and cur == 1:
                counter += 1
            prev = cur
        return counter

    # since we know no points are at edge of matrix (previous assertions) loop through all non-edge points.
    DIM_Y = len(im)
    DIM_X = len(im[0])
    for i in range(1, DIM_Y - 2):
        for j in range(1, DIM_X - 2):
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


ITERATIONS = 10000
profiling.begin("peter")
for i in range(ITERATIONS):
    result_peter = peter_calculate_branch_and_endpoints(EXAMPLE_SPLINE)
profiling.end("peter")

profiling.begin("heltena")
for i in range(ITERATIONS):
    result_heltena = heltena_calculate_branch_and_endpoints(EXAMPLE_SPLINE)
profiling.end("heltena")

if result_peter != result_heltena:
    print "ERROR"
    print "peter   ", result_peter
    print "heltena ", result_heltena