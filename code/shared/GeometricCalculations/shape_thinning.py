import numpy as np
'''
def calculate_new_P(P, direction):
    def check1(P):
        counter = 0
        for i in xrange(1, 8):
            if P[i] == 0 and P[i + 1] == 1: counter += 1
        if P[8] == 0 and P[1] == 1: counter += 1
        if counter == 1:
            return True
        else:
            return False

    def check2(P):
        if 2 <= sum(P[1:]) <= 6:
            return True
        else:
            return False

    def check3(P, direction):
        if direction == 'east, south':
            if P[1] * P[3] * P[5] == 0 and P[3] * P[5] * P[7] == 0:
                return True
            else:
                return False
        else:
            if P[1] * P[3] * P[7] == 0 and P[1] * P[5] * P[7] == 0:
                return True
            else:
                return False

    action = ''
    if check1(P):
        if 2 <= sum(P[1:]) <= 6:
            if check3(P, direction):
                action = 'remove ' + direction
                return 1, action
            else:
                action = 'not C ' + direction
        else:
            action = 'not B'
    else:
        action = 'not A'
    return 0, action

def calculate_M_old(image_array, direction):
    im = image_array
    M = np.zeros([len(im), len(im[0])], dtype=int)
    for i in xrange(1, len(im) - 1):
        for j in xrange(1, len(im[0]) - 1):
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

                M[i][j], action = calculate_new_P(P, direction)
            else:
                M[i][j] = 0

                #print P[8], P[1], P[2]
                #print P[7], P[0], P[3]
                #print P[6], P[5], P[4]

                #print M[i][j], action
                #raw_input('press_key')
    return M

def calculate_M(image_array, direction):
    im = image_array
    M = np.zeros([len(im), len(im[0])], dtype=int)
    for i, im_row in enumerate(im[1:-1], start=1):
        for j, _ in enumerate(im_row[1:-1], start=1):
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
                M[i][j], action = calculate_new_P(P, direction)
            else:
                M[i][j] = 0
    return M
'''
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
'''
def write_stuff(im, M, direction):
    print 'im1', direction
    print im
    print 'm', direction
    print M

def skeletonize(image_array):
    """
    :param image_array:
    :return:
    """
    im_array = add_border(image_array)
    counter = 0
    pixelRemoved = True
    while pixelRemoved:
        pixelRemoved = False
        counter += 1

        M = calculate_M(im_array, 'east, south')
        #M1 = calculate_M_old(im_array, 'east, south')
        if M.any() == 1: pixelRemoved = True
        #write_stuff(im_array, M, 'east, south')
        #write_stuff(M, M1, 'east, south')
        im_array = im_array - M

        M = calculate_M(im_array, 'north, west')
        #M1 = calculate_M_old(im_array, 'north, west')
        if M.any() == 1: pixelRemoved = True
        #write_stuff(im_array, M, 'north, west')
        im_array = im_array - M

        if counter >= 1000: pixelRemoved = False

    thinned_image = remove_border(im_array)
    return thinned_image
'''
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
