import numpy as np

#
# def add_border(im):
#     new_image = np.zeros([len(im) + 2, len(im[0]) + 2], dtype=int)
#     for i, im_row in enumerate(im):
#         for j, _ in enumerate(im_row):
#             new_image[i + 1][j + 1] = im[i][j]
#     return new_image
#
# def remove_border(im):
#     new_image = np.zeros([len(im) - 2, len(im[0]) - 2], dtype=int)
#     for i, im_row in enumerate(im[1:-1], start=1):
#         for j, _ in enumerate(im_row[1:-1], start=1):
#             new_image[i - 1][j - 1] = im[i][j]
#     return new_image

def add_border(im):
    new_image = np.zeros([len(im) + 2, len(im[0]) + 2], dtype=int)
    new_image[1:len(im)+1,1:len(im[0])+1] = im
    return new_image

def remove_border(im):
    return im[1:-1,1:-1]

def iterate_z(Z, subiteration=0):
    # count neighbors

    #HELTENA removed, new code below
    # N = np.zeros(Z.shape, int)
    # N[1:-1,1:-1] += (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
    #                  Z[1:-1,0:-2]                + Z[1:-1,2:] +
    #                  Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
    # # if Neighbors between 2 and 6, consider for removal
    # check1 = (2 <= N) & (N <= 6)
    # # count 01 edges, E
    # E = np.zeros(Z.shape, int)
    # p = [Z[0:-2,1:-1], Z[0:-2,2:], Z[1:-1,2:], Z[2:  ,2:],
    #      Z[2:  ,1:-1], Z[2:  ,0:-2], Z[1:-1,0:-2], Z[0:-2,0:-2],
    #      Z[0:-2,1:-1]]

    p = [Z[0:-2, 1:-1], Z[0:-2, 2:], Z[1:-1, 2:], Z[2:, 2:],
         Z[2:  , 1:-1], Z[2:  , 0:-2], Z[1:-1, 0:-2], Z[0:-2, 0:-2]]
    N = np.zeros(Z.shape, int)
    N[1:-1, 1:-1] = sum(p)
    check1 = (2 <= N) & (N <= 6)

    # count 01 edges, E
    E = np.zeros(Z.shape, int)
    p.append(p[0])

    pold = p[0]
    for pi in p[1:]:
        E[1:-1, 1:-1] += (pold == 0) & (pi == 1)
        pold = pi

    #HELTENA removed, no indexing array.
    #for i, pi in enumerate(p[:-1]):
    #    E[1:-1,1:-1] += (pi == 0) & (p[i+1] == 1)

    # if edge count ==1 consider for removal
    check2 = (E == 1)

    #HELTENA removed, we can use the defined above, offset -2!
    #p = [0, 0, Z[0:-2,1:-1], Z[0:-2,2:], Z[1:-1,2:], Z[2:  ,2:],
    #     Z[2:  ,1:-1], Z[2:  ,0:-2], Z[1:-1,0:-2], Z[0:-2,0:-2]]

    if subiteration == 0:
        p24 = p[2] * p[4]
        east_wind = np.zeros(Z.shape, int)
        #east_wind[1:-1,1:-1] = Z[ :-2,1:-1] * Z[1:-1,2:] * Z[2:  ,1:-1]
        #east_wind[1:-1,1:-1] = p[2] *p[4] * p[6]
        east_wind[1:-1,1:-1] = p[0] * p24 # p[2] * p[4] # offset p[-2]!!
        check3 = (east_wind == 0)

        south_wind = np.zeros(Z.shape, int)
        #south_wind[1:-1,1:-1] = Z[1:-1,2:] * Z[2:  ,1:-1] * Z[1:-1, :-2]
        #south_wind[1:-1,1:-1] = p[4] * p[6] * p[8]
        south_wind[1:-1,1:-1] = p24 * p[6] #p[2] * p[4] * p[6] # offset p[-2]!!
        check4 = (south_wind == 0)
    else:
        p06 = p[0] * p[6]
        west_wind = np.zeros(Z.shape, int)
        #west_wind[1:-1,1:-1] = Z[ :-2,1:-1] * Z[1:-1,2:] * Z[1:-1, :-2]
        #west_wind[1:-1,1:-1] = p[2] *p[4] * p[8]
        west_wind[1:-1,1:-1] = p06 * p[2] # p[0] *p[2] * p[6] # offset p[-2]!!
        check3 = (west_wind == 0)

        north_wind = np.zeros(Z.shape, int)
        #north_wind[1:-1,1:-1] = Z[ :-2,1:-1] * Z[2:  ,1:-1] * Z[1:-1, :-2]
        #north_wind[1:-1,1:-1] = p[2] * p[6] * p[8]
        north_wind[1:-1,1:-1] = p06 * p[4] # p[0] * p[4] * p[6] # offset p[-2]!!
        check4 = (north_wind == 0)
        
    removal = check1 & check2 & check3 & check4
    Z1 = (Z == 1)
    removed = Z1 & (removal == 1)
    points_removed = removed.any()
    Z = np.array(Z1 & (removal == 0), int)
    return Z, points_removed
        
def skeletonize(im_array):
    """
    :param image_array:
    :return:
    """

    Z = add_border(im_array)
    for i in range(1000):
        Z, points_removed1 = iterate_z(Z, subiteration=0)
        Z, points_removed2 = iterate_z(Z, subiteration=1)        
        if not points_removed1 and not points_removed2:
            break
    else:
        print 'manually stopped thinning algorithm at 1000 interations'
    thinned_image = remove_border(Z)
    return thinned_image

if __name__ == '__main__':
    '''    
    im_array = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    '''
    im_array = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0]])


    #print im_array
    #profiling.begin()
    for i in range(10000):
        skim = skeletonize(im_array)
    #profiling.end()
    #print 'final'
    print skim
    #calculate_M(im_array, 'first')
    #im = Image.new('1',[x_range, y_range])
    #im.show()
