import math

def compute_angles4(x1, y1, x2, y2):
    dx=x2-x1
    dy=y2-y1
    angle= math.atan2(dy,dx)/math.pi
    
    if(math.fabs(angle)>1):
        print "wrong angle ", angle
    return angle


def angles_to_shapes(angles,d,worm_angle_average):
    
    """
        d is the original pairwise vector (singol snapshot)
        d are raw data
        angles are not standardized angles (variance of angle1, angle2 is NOT 1)
        but they can have average 0, because this is added in worm_angle_average
    """
    
    d_reconstruct=[]
    for k in range(0,len(d)-2,2):
        dx=d[k+2]-d[k]
        dy=d[k+3]-d[k+1]
        
        dl=math.sqrt(dx*dx+dy*dy)
        #print dl
        angle=angles[k/2]+worm_angle_average
        dx2=dl*math.cos(angle*math.pi)
        dy2=dl*math.sin(angle*math.pi)
        #print "worm angle ", k, worm_angle_average, angle
        #print "error", dy-dy2, dx-dx2
        
        if k==0:
            d_reconstruct.append(d[0])
            d_reconstruct.append(d[1])
        
        previuos_x=d_reconstruct[-2]
        previuos_y=d_reconstruct[-1]
        d_reconstruct.append(previuos_x+dx2)
        d_reconstruct.append(previuos_y+dy2)
    
    return d_reconstruct


def check_curve_is_smooth(angle_w):
    
   # new_angles=[]
   # if len(angle_w)==0:
   #    return new_angles
    
   # new_angles.append(angle_w[0])
    for i in range(1, len(angle_w)):
        if math.fabs(angle_w[i]-angle_w[i-1])>1.75:
            angle_w[i]+=2.
        if math.fabs(angle_w[i]-angle_w[i-1])>1.75:
            angle_w[i]-=4.
    #new_angles.append(angle_w[i])

def shapes_to_angles(snapshots, zero_average):
    
    """
        this functions computes the angles from all the original snapshot shapes
        zero_average is a boolean to decise if you want the angles to have zero average
    """
    angles=[]
    for w in snapshots:
        if len(w) >0: assert type(w[0]) not in [list, tuple], 'you are using the new format. this function likes the old one'
        angle_w=[]
        for k in range(0,len(w)-2,2):
           angle_w.append(compute_angles4(w[k], w[k+1], w[k+2], w[k+3]))
           
        check_curve_is_smooth(angle_w)
        
        if(zero_average):
            set_zero_average(angle_w)
        
        angles.append(angle_w)
    
    
    return angles


def set_zero_average(a):
    av=sum(a)/len(a)
    for k, b in enumerate(a):
        a[k]-=av


def from_tuple_to_pairwise(s):

    """
        s in a list of tuples
        return a pairwise vector [x1, y1, x2, y2, ...]
    """
    l=[]
    for x,y in s:
        l.append(x)
        l.append(y)
    return l