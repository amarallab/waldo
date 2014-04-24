import math

import numpy as np


def average_and_std_of_a_list_sample(l):
    
    sum=0
    sum_of_squares=0    
    for v in l:
        sum+=v
        sum_of_squares+=v*v
    var=float( sum_of_squares) /len(l)  - float(sum)/len(l) * float(sum)/len(l)
    var=var*float(len(l)) / float(len(l)-1)
    if var<0:
        var=0
    return ( float(sum)/len(l) , math.sqrt(var)  )            


def compute_transpose(x):
    """ given a matrix, computes the transpose """
    xt=[([0]*len(x)) for k in x[0]]
    for i, x_row in enumerate(x):
        for j, b in enumerate(x_row):
            xt[j][i]=x[i][j]
    return xt

def standardize_matrix(x):
    """
        it computes the average of each column and standard dev
        and standardize the columns
        return lists of tuples(av, std) for each variable
    """
    xt=compute_transpose(x)
    av_stds=[average_and_std_of_a_list_sample(v) for v in xt]
    for i, x_row in enumerate(x):
        for j, b in enumerate(x_row):
            x[i][j]-=av_stds[j][0]
            x[i][j]/=av_stds[j][1]
    return av_stds

def euclidean_distance(x1, y1, x2, y2):
    
    dx=x2-x1
    dy=y2-y1
    #print dx, dy, math.sqrt(dx*dx+dy*dy)
    return math.sqrt(dx*dx+dy*dy)


def compute_euclidean_distance(x, xnext):
    
    """
        this function computes the euclidean distance of this two vectors        
        """
    dist=0.
    for k, a in xrange(len(x)):
        dist+=(x[k]-xnext[k])*(x[k]-xnext[k])
    return math.sqrt(dist)



def dot_product(x,x2):
    return np.dot(np.array(x),np.array(x2)) 

def cosine_distance(x1,x2):
    norm1 = np.dot(x1,x1)
    norm2 = np.dot(x2, x2)
    return np.dot(x1, x2)/math.sqrt(norm1*norm2)


def euclidean_cosine_distances(approx_data, data):

    distances=[]
    for k in xrange(len(approx_data)):
        distances.append([compute_euclidean_distance(approx_data[k], data[k]),
                          cosine_distance(approx_data[k], data[k])])
    return distances

def print_points(x, filename):
    f=open(filename, "w")
    for k in xrange(0,len(x),2):
        f.write(str(x[k])+" "+str(x[k+1])+"\n")
