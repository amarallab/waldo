import numpy as np

def add_border(im):
    new_image = np.zeros([len(im) + 2, len(im[0]) + 2], dtype=int)
    for i, im_row in enumerate(im):
        for j, _ in enumerate(im_row):
            new_image[i + 1][j + 1] = im[i][j]
    return new_image

def remove_border(im):
    new_image = np.zeros([len(im) - 2, len(im[0]) - 2], dtype=int)
    for i, im_row in enumerate(im[1:-1], start=1):
        for j, _ in enumerate(im_row[1:-1], start=1):
            new_image[i - 1][j - 1] = im[i][j]
    return new_image

def iterate_z(Z, subiteration=0):
    # count neighbors
    N = np.zeros(Z.shape, int)
    N[1:-1,1:-1] += (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
                     Z[1:-1,0:-2]                + Z[1:-1,2:] +
                     Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
    # if Neighbors between 2 and 6, consider for removal
    check1 = (2 <= N) & (N <= 6)
    # count 01 edges, E
    E = np.zeros(Z.shape, int)    
    p = [Z[0:-2,1:-1], Z[0:-2,2:], Z[1:-1,2:], Z[2:  ,2:],
         Z[2:  ,1:-1], Z[2:  ,0:-2], Z[1:-1,0:-2], Z[0:-2,0:-2],
         Z[0:-2,1:-1]]
    for i, pi in enumerate(p[:-1]):
        E[1:-1,1:-1] += (pi == 0)  & (p[i+1] == 1)
    # if edge count ==1 consider for removal
    check2 = (E == 1)

    p = [0, 0, Z[0:-2,1:-1], Z[0:-2,2:], Z[1:-1,2:], Z[2:  ,2:],
         Z[2:  ,1:-1], Z[2:  ,0:-2], Z[1:-1,0:-2], Z[0:-2,0:-2]]
             
    if subiteration == 0:
        east_wind = np.zeros(Z.shape, int)
        #east_wind[1:-1,1:-1] = Z[ :-2,1:-1] * Z[1:-1,2:] * Z[2:  ,1:-1]
        east_wind[1:-1,1:-1] = p[2] *p[4] * p[6]
        check3 = (east_wind ==0)

        south_wind = np.zeros(Z.shape, int)
        #south_wind[1:-1,1:-1] = Z[1:-1,2:] * Z[2:  ,1:-1] * Z[1:-1, :-2]
        south_wind[1:-1,1:-1] = p[4] * p[6] * p[8]
        check4 = (south_wind ==0)
    else:
        west_wind = np.zeros(Z.shape, int)
        #west_wind[1:-1,1:-1] = Z[ :-2,1:-1] * Z[1:-1,2:] * Z[1:-1, :-2]
        west_wind[1:-1,1:-1] = p[2] *p[4] * p[8]
        check3 = (west_wind ==0)

        north_wind = np.zeros(Z.shape, int)
        #north_wind[1:-1,1:-1] = Z[ :-2,1:-1] * Z[2:  ,1:-1] * Z[1:-1, :-2]
        north_wind[1:-1,1:-1] = p[2] * p[6] * p[8]
        check4 = (north_wind ==0)

        
    removal = check1 & check2 & check3 & check4
    removed = (Z == 1) & (removal == 1)
    points_removed = removed.any()
    Z = np.array((Z == 1) & (removal == 0), int)
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
    skim = skeletonize(im_array)
    #print 'final'
    print skim
    #calculate_M(im_array, 'first')
    #im = Image.new('1',[x_range, y_range])
    #im.show()
