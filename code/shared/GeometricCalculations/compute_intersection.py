import math

def intersection_point_between_lines(p1, p2, m, q1):
    '''
        p1 and p2 are two points between we draw a line
        the other line is specified by the angular coefficient 
        and one point q1
    '''
    
    # p1=list(p1)
    # p2=list(p2)
    # if p1[0] == p2[0]:
    #     p1[0] += 1e-6
    #
    # n = (p2[1]-p1[1]) / (p2[0]-p1[0])
    #
    # if n == m:
    #     n += 1e-6
    #
    # intersection_x = (p1[1] - q1[1] + m * q1[0] -n * p1[0]) / (m-n)
    # intersection_y = q1[1] + m * (intersection_x - q1[0])
    # return intersection_x, intersection_y

    x1, y1 = p1
    x2, y2 = p2
    xq, yq = q1
    if x1 == x2:
        x1 += 1e-6
    n = (y2-y1) / (x2-x1)
    if n == m:
        n += 1e-6
    intersection_x = (y1 - yq + m * xq - n * x1) / (m - n)
    intersection_y = yq + m * (intersection_x - xq)
    return intersection_x, intersection_y

# def peter_points_along_straight_line(m, p1):
#     xs = []
#     ys = []
#     for k in [-0.2, 0.2]:
#         xs.append(p1[0] + k)
#         ys.append(p1[1] + m * (xs[-1]-p1[0]))
#     return xs, ys
    
def points_along_straight_line(m, p1):
    x, y = p1
    xs = [x - 0.2, x + 0.2]
    ys = [y - m * 0.2, y + m * 0.2]
    return xs, ys

def get_ortogonal_to_spine(spine, index):
    '''
        spine is a list of [x,y]
        index is the index in this list from witch we compute the
        perpendicular line to the midpoint
    '''
    p1 = spine[index]
    if index + 1 != len(spine):
        p2 = spine[index+1]
    else:
        p2 = p1
        p1 = spine[index-1]
    # this is simplify things - although it introduces a very small error -
    # but I believe it is negligible
    p1, p2 = list(p1), list(p2)
    if p1[0] == p2[0]:
        p1[0] += 1e-6
    m= (p2[1]-p1[1])/(p2[0]-p1[0])
    #print m, 'angular coefficient'
    if m == 0:
        m = 1e-6
    return -1./m, p1
    
def find_intersection_points(q1, m, outline):
    xs, ys = [], []
    p1 = outline[0]
    for p2 in outline[1:]:
        ip = intersection_point_between_lines(p1, p2, m, q1)
        #acceptable_error = 1e-6
        if  ip[0]+1e-6 >= min(p1[0], p2[0]) and ip[0]-1e-6 <= max(p1[0], p2[0]) and \
            ip[1]+1e-6 >= min(p1[1], p2[1]) and ip[1]-1e-6 <= max(p1[1], p2[1]):
            xs.append(ip[0])
            ys.append(ip[1])
        p1 = p2
    return xs, ys

#HELTENA: this method doesn't return area. Use the next method.
def check_point_is_inside_box(q1, p1, p2, acceptable_error=1e-6):
    # area= math.fabs((p2[1] - p1[1]) * (p2[0] - p1[0]))
    return (min(p1[0], p2[0])-acceptable_error) <= q1[0] <= (max(p1[0], p2[0])+acceptable_error) and \
            (min(p1[1], p2[1]) - acceptable_error) <= q1[1] <= (max(p1[1], p2[1])+acceptable_error)

def calculate_area_of_box(p1, p2):
    return math.fabs((p2[1] - p1[1]) * (p2[0] - p1[0]))


