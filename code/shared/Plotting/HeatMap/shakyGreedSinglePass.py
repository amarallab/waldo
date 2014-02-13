
# Takes a matrix from stdin pipe
# and reorders it using a greedy algorithm
#
#
# THE INPUT IS A LINK LIST i j 5
#
# THIS ASSUMES DIRECTIONAL NETWORK
# if this encounters i j 3, it doesn't
# automatically assume that j i 3
# exists as well.
#


'''
    Example:
    cat TESTquant.ll | python shakyGreedSinglePass.py 854 > ordered-TESTquant.ll
    mv transtable_shakyGreedy.dat orderIndices-TESTquant.dat

'''


import sys


# PARAMETERS
from Shared.Code.Plotting.HeatMap import progress_bar, dictmatrix

try:
    size = int(sys.argv[1])
except IndexError:
    msg = """

Usage:
< inputlinklist.ll  python shakyGreed.py <SIZE>  > outputlinklist.ll

You need to give size of the matrix

"""
    raise IOError(msg)





search_depth = 1  #100            # 1 -- 10000 [higher gives higher accuracy, slower speed]
seed = 1111                   # random number generator seed
plot = False                  # True or False. True: plot both unordered and ordered matrices
print_transtable = True       # True or False. True: put the transtable into a transtable.dat
calculate_cost = False        # True or False. True: report on the cost(energy) of ordered mx.


if search_depth > 1:
    calculate_cost = True


# the following have good default
# values for any application (6 and 0.15)
weak_shuffle_windowsize = 6   # 4 -- 100  
resort_frequency = 0.15       # 0.01 -- 1 (portion of steps to do resorting)
weighted = True


#---------------------------------------------

import random as rnd

mx = dictmatrix(0.)
# INPUT -- WORD COUNT
# links = []
# p = progress_bar(nlinks)
# for line in sys.stdin:
#     p.next()
#     i, jw = line.split('_')
#     j,w = jw.rstrip().split(':')
#     i,j,w = map(int, [i,j,w])
#     i -=1                       #i-1 j-1 CAREFUL (node labels start from 1 not 0)
#     j -=1
#     links.append( (i,j,w) )  
#     mx.put(i,j,w)       
# print >> sys.stderr, 'input read.'

# INPUT -- LINKLIST
links = []
for line in sys.stdin:
    i,j,w = line.split()
    i,j = map(int, [i,j])
    #i-=1
    #j-=1
    w = float(w)
    if w != 0:
        links.append( (i,j,w) )  
        if calculate_cost:
            mx.put(i,j,w)
print >> sys.stderr, 'input read.'




# inform size
nlinks = len(links)
print >> sys.stderr, "<shakyGreed> Network size: %i nodes" % size
print >> sys.stderr, "<shakyGreed> Network size: %i links" % nlinks
print >> sys.stderr, "<shakyGreed> Network sparsity: %.4f%% of possible edges" % (100.*nlinks/(size*size))


# COST CALCULATION
# needs mx and size to be declared
def cost(seq, mx, size, verbose=False):
    if verbose:
        print >> sys.stderr, 'cost calculation'
    C = 0.
    if len(seq) != size:
        print >>sys.stderr, seq
        print >>sys.stderr, len(seq), size, '!!!!!!!!!!!!!!'
        raise IndexError('Sequence length != matrix size')
    p = progress_bar(size*size)
    for i in xrange(size):
        for j in xrange(size):
            if verbose:
                p.next()
            d = abs(i-j)
            i_ = seq[i]
            j_ = seq[j]
            C += mx[i_][j_] * d
    return C


# WEAK SHUFFLE
def weak_shuffle(lst, windowsize, overlap=2):
    newlst = []
    if windowsize <3:
        raise ValueError("No weak shuffle with a window smaller than 3")
    n_windows = (len(lst)-overlap)//(windowsize-overlap)
    # first move
    window = lst[:windowsize]
    rnd.shuffle(window)
    newlst = window
    # loop - shuffle every window
    for k in xrange(1,n_windows):
        window = window[-overlap:] + lst[(overlap+k*(windowsize-overlap)):(overlap+((k+1)*(windowsize-overlap)))]
        rnd.shuffle(window)
        newlst = newlst[:-overlap] + window
    # last move
    window = window[-overlap:] + lst[(overlap+(k+1)*(windowsize-overlap)):]
    rnd.shuffle(window)
    newlst = newlst[:-overlap] + window
    # done!
    return newlst


# DEFINE DATA STRUCTURE
class kernellist(list):
    def __init__(self):
        list.__init__(self)


    # DEFINE MOVES
    def create_new_kernel(self, i,j):
        """
        create a new kernel with i and j together
        """
        self.append( [i,j] )

    def add_single(self, i, kernelj, j):
        """
        add a single i to a kernel
        add it to the side closer to j in it
        """
        posj = kernelj.index(j)
        if posj < len(kernelj)/2:
            kernelj.insert(0, i)
        else:
            kernelj.append(i)

    def bind_kernels_together(self, i, kerneli, j, kernelj):
        """
        bind two kernels together
        position them so that distance between
        i and j is minimum
        """
        # kernelj to left or right of kerneli
        posi = kerneli.index(i)
        if posi < len(kerneli)/2:
            anchorside = 'left'
        else:
            anchorside = 'right'
        # flip kernelj or not
        posj = kernelj.index(j)
        if (anchorside=='left' and posj < len(kernelj)/2) or \
                (anchorside=='right' and posj >= len(kernelj)/2):
            # flip
            kernelj.reverse()
        else:
            # no flip
            pass
        # One ring to bring them all
        # and in the darkness bind them
        kernelino = kernels.index(kerneli)
        kerneljno = kernels.index(kernelj)
        if anchorside == 'left':
            self[kernelino] = kernelj + kerneli
        elif anchorside == 'right':
            self[kernelino] = kerneli + kernelj
        else:
            raise KeyError("Warning! Rapture in space-time!")
        # now the kernel with kernelino is the new, bound double-kernel
        # delete the old kernelj from memory
        del self[kerneljno]

    # FINAL SEQUENCE
    def sequence(self):
        # append all floating kernels together
        # and return them as one single list
        return reduce(lambda l1,l2: l1+l2, self)





# ALGORITHM

print >> sys.stderr, 'ShakyGreed Matrix Ordering'

# apply parameters
steps = int(search_depth * 10000./len(links))
if steps < 10:
    steps = 10
if search_depth == 1:
    steps = 1
print >> sys.stderr, 'run for %i steps' % steps

if resort_frequency > 1.:
    resort_frequency = 1.
resort_steps = int(steps // (resort_frequency*search_depth)) + 1
if weighted and resort_steps < steps:
    print >> sys.stderr, 're-sort every %i steps' % resort_steps

rnd.seed(seed)

# initialize
all_seq = []
if search_depth > 1:
    pbar = progress_bar(steps, timestep=60)

# loop over (weakly) shuffled orderings of links
# (shaky part of shakyGreed)
for k in xrange(steps):


    # initialize
    kernels = kernellist()

    if weighted:
        # re-sort links before weak shuffle
        if k % resort_steps == 0:
            links.sort(key=lambda (i,j,w): w, reverse=True)
        # weak shuffle to search for better sequences
        if k != 0:
            links = weak_shuffle(links, weak_shuffle_windowsize)

    elif not weighted:
        # shuffle link ordering
        # {no weak-shuffle, regular strong shufle}
        rnd.shuffle(links)

    # greedy algorithm for a given ordering of links
    # (greed part of shakyGreed)
    if search_depth == 1:
        pp = progress_bar(len(links))
    for i,j,w in links:
        if i == j:
            continue
        #print i,j
        # don't really need w for this algorithm,
        # it was just there to sort the links
        kerneli, kernelj = None, None
        i_found, j_found = False, False
        # seek i and j in kernels
        for kernel in kernels:
            if (not i_found) and (i in kernel):
                i_found = True
                kerneli = kernel
            if (not j_found) and (j in kernel):
                j_found = True
                kernelj = kernel
            if i_found and j_found:
                break
        # choose move according to results
        if (not i_found) and (not j_found):
            # neither were in a kernel. 
            # create new kernel.
            kernels.create_new_kernel(i,j)
            #print 'create new kernel'
            #print kernels
        elif i_found and not j_found:
            # only i was found in a kernel
            # add j as a single to kerneli
            kernels.add_single(j, kerneli, i) 
            #print 'add single %i to kernel with %i' % (j,i)
            #print kernels
        elif j_found and not i_found:
            # only j was found in a kernel
            # add i as a single to kernelj
            kernels.add_single(i, kernelj, j)
            #print 'add single %i to kernel with %i' % (i,j)
            #print kernels
        elif i_found and j_found:
            # both were encountered
            if kerneli == kernelj:
                # both were in the same kernel
                # do nothing
                pass
            else:
                # each was in a different kernel
                # bind these kernels together
                kernels.bind_kernels_together(i, kerneli, j, kernelj)
                #print 'bind kernels of %i and %i' % (i,j)
                #print kernels

        if search_depth == 1 and nlinks > 1e6:
            pp.next()



    # Went through all the links
    # Final sequence is built
    seq = kernels.sequence()
    for node in xrange(size):
        if node not in seq:
            # this node didn't have any links
            seq.append(node)

    # report on progress and time left
    if search_depth > 1 and steps * nlinks > 1e6:
        pbar.next()
    if search_depth == 1 or not calculate_cost:
        COST = 1.
    else:
        COST = cost(seq, mx, size)

    all_seq.append( (COST,seq) )


# --print sequences (debug)--
# all_seq.sort()
# for s in all_seq:
#     print >> sys.stderr, s

# choose min cost
C, seq = min(all_seq)

if calculate_cost:
    #print >> sys.stderr, 'initial (unordered) cost:', cost(range(size), mx, size)
    print >> sys.stderr, 'first pass cost:', all_seq[0][0]
    print >> sys.stderr, 'found minimum cost:', C


# BUILD THE ORDERED MATRIX (AS A LINKLIST)
# ACCORDING TO THE FOUND SEQUENCE
# AND PRINT IT
if nlinks > 1e6: 
    print >> sys.stderr,'---writing out the ordered matrix---'
transtable = dict(zip(seq, range(size)))
pp = progress_bar(nlinks)
for i,j,w in links:
    if nlinks > 1e6: 
        pp.next()
    print '%i %i %g' % (transtable[i], transtable[j], w)



#PRINT THE TRANSTABLE
#{NEW POS: OLD POS}
if print_transtable:
    filename = 'transtable_shakyGreedy.dat'
    f = open(filename,'w')
    for i in xrange(size):
        print >>f, i, seq[i]
    f.close()
    print >> sys.stderr, filename, 'written.'

# BUILD THE ORDERED MATRIX (DICTMATRIX)
# ACCORDING TO THE FOUND SEQUENCE
# AND PRINT IT
# ordered_mx = dictmatrix(0.)
# #p = progress_bar(size*size)
# for i in range(size):
#     for j in range(size):
#         #p.next()
#         iold = seq[i]
#         jold = seq[j]
#         ordered_mx[i][j] = mx[iold][jold]
#         print '%.1f' % mx[iold][jold],
#     print



