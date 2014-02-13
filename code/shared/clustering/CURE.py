# CURE.py
#
# CURE Clustering Algorithm
#
#

import numpy as np
from scipy.spatial.kdtree import KDTree
from scipy.spatial.distance import cosine, euclidean, correlation, sqeuclidean
from scipy.spatial.distance import cdist, pdist
import heapq

distFunc = {'euclidean':euclidean,
            'cosine':cosine,
            'correlation':correlation,
            'sqeuclidean':sqeuclidean}

DISTANCE_METRIC='cosine'
#'correlation'

INF = float('inf')

class HeapError(Exception):
    pass


def smooth_float_rounding_errors(result):
    epsilon = 1e-13
    if -epsilon < result < 0:
        return 0
    else:
        return result

def dist_point(point1, point2, metric=DISTANCE_METRIC):
    result = distFunc[metric](point1, point2)
    return smooth_float_rounding_errors(result)

def distance(cluster1, cluster2):
    pairwise = cdist(cluster1.representatives, cluster2.representatives, metric=DISTANCE_METRIC)
    result =  np.average(pairwise)
    return smooth_float_rounding_errors(result)

class Cluster():

    # def __new__(self, *args, **kwargs):
    #     obj = np.ndarray.__new__(self, *args, **kwargs)
    #     obj.test = 'hey'
    #     return obj

    nextId = 0
    
    def __init__(self, point=None):
        self.points = np.array([point])
        self.numRep = 3
        self.centroid = np.array([point])
        self.alpha = 0.5
        self.representatives = np.array([point])
        self.closest = None
        self.closest_dist = 0

        self.id = Cluster.nextId
        Cluster.nextId += 1


    def __repr__(self):
        return str(self.id)
    

    def distanceToCluster(self, other):
        pairwiseDistances = cdist(self.points, other.points, metric=DISTANCE_METRIC)
        # 'average distance'
        return np.average(pairwiseDistances)

    def calc_mean(self):
        self.centroid = self.points.mean(axis=0)


    def calc_representatives(self):

        self.representatives = []

        for i in xrange(self.numRep):
            maxDist = 0
            for point in self.points:
                if i == 0:
                    minDist = dist_point(point, self.centroid)
                else:
                    distToq = lambda q: dist_point(point, q)
                    distancesToReps = map(distToq, self.representatives) 
                    minDist = np.min(distancesToReps)
                    
                if minDist >= maxDist:
                    maxDist = minDist
                    maxPoint = point

            self.representatives.append(maxPoint)

        self.representatives = np.array(self.representatives)


    def __lt__(self, other):
        return self.closest_dist < other.closest_dist

    def __gr__(self, other):
        return self.closest_dist > other.closest_dist
    # def __eq__(self, other):
    #     return self.closest_dist == other.closest_dist
    # def __ne__(self, other):
    #     return self.closest_dist != other.closest_dist



def merge(cluster1, cluster2):
    merged = Cluster()
    merged.points = np.vstack( (cluster1.points,cluster2.points) )
    merged.calc_mean()
    merged.calc_representatives()
    # shrink
    tmpSet = merged.representatives
    merged.representatives = []
    for point in tmpSet:
        merged.representatives.append(  point + merged.alpha*(merged.centroid - point)    )
    return merged



def remove_from_heap(heap, item):
    heap.remove(item)
    heapq.heapify(heap)
    if item in heap:
        print heap
        raise HeapError("Could not remove %s" % item.id)

    return heap



class CURE():
    
    def cluster(self, inputVector, stop_value, stop_criterion='num_clusters'):
        """
           Cluster using CURE. Distance metric is set at the beginning of the code

           Input:
           inputVector: ndarray with points. Example: [[2,3], [3,1], [5,2]]
           stop_value & stop_criterion: 
                             If the criterion is "num_clusters", stop_value is that number
                             If the criterion is "distance_threshold", stop_value is that threshold

                  
           Output:
           Clusters. A simple list. One int per point.
           n'th item is the cluster id of the n'th point in inputVector
        """
        


        if stop_criterion == 'distance_threshold':
            def stop_criteria_reached(distance_to_merge):
                return distance_to_merge > stop_value or len(self.heap) <= 1
        elif stop_criterion == 'num_clusters':
            def stop_criteria_reached(distance_to_merge):
                return len(self.heap) <= max(stop_value, 1)
        else:
            raise ValueError('stop_criterion is either "num_clusters" or "distance_threshold"')
        
        
        #self.numClusters = numClusters
        # clusterID --> clusterObject
        #self.id2Cluster = {}
        # clusterReps --> clusterID
        #self.reps2clusterID = {}

        # distances between each point
        #distances = cdist(inputVector, inputVector, metric=DISTANCE_METRIC)

        # Step 1---------------
        # each point is its own cluster at first
        clusters = []
        for i, point in enumerate(inputVector):
            cluster = Cluster( point )
            #self.id2Cluster[i] = cluster
            #self.reps2clusterID[cluster.representatives] = i
            clusters.append( cluster )
 



        # find the closest clusters
        for i, cluster_i in enumerate(clusters):
            cluster_i.closest_dist = INF
            cluster_i.closest = None
            for j, cluster_j in enumerate(clusters):
                if i != j:
                    dist_ij = distance(cluster_i, cluster_j)
                    if  dist_ij < cluster_i.closest_dist:
                        cluster_i.closest_dist = dist_ij
                        cluster_i.closest = cluster_j
                    
            # myDistances = distances[i]
            # # self-distace set to inf to avoid picking yourself as the closest
            # myDistances[i] = INF 
            # # find closest cluster (also record the distance
            # closestInd = np.argmin(myDistances)
            # cluster.closest_dist = np.min(myDistances)
            # cluster.closest = clusters[closestInd]


        # clear memory
        #del distances



        # Step 3--------------
        # put clusters in a heap
        self.heap = []
        for cluster in clusters:
            heapq.heappush(self.heap, cluster)
        

        # Step 4-----------
        # While loop
        distance_to_merge = 0.
        while not stop_criteria_reached(distance_to_merge):

            # Step 5-----------
            # merge the two closest clusters
            u = heapq.heappop(self.heap)
            v = u.closest
            distance_to_merge = u.closest_dist
            # print 'u', u.id, 'v', v.id
            # print 'distances', u.closest_dist, v.closest_dist, distance(u,v)
            # print 'closest ids', u.closest.id, v.closest.id
            # print 'min of heap', self.heap[0].id, 'dist', self.heap[0].closest_dist, 'to', self.heap[0].closest.id
            # print 'v in heap?', v in self.heap
            self.heap = remove_from_heap(self.heap, v)
            # print 'after removing, u in heap?', u in self.heap, 'v in heap?', v in self.heap

            # sanity check, and remove v from the heap (v is the next min
            # sanity_check = heapq.heappop(self.heap)
            # print 'heappop result', sanity_check, sanity_check.closest_dist
            # if v != sanity_check:
            #     raise HeapError
            
            w = merge(u,v)

            
            # remove u & v from id2Cluster and reps2clusterID
            # uID = reps2clusterID[u.representatives]
            # vID = reps2clusterID[u.representatives]
            # del id2Cluster[uID]
            # del id2Cluster[vID]
            # del reps2clusterID[u.representatives]
            # del reps2clusterID[v.representatives]
            # # add w to id2Cluster and reps2clusterID
            # wID = uID
            # self.reps2clusterID[w.representatives] = wID
            # self.id2Cluster[wId] = w
            # print w


            # # calculate new distances from w to the others
            # w.closest_dist = INF
            # for cluster in self.heap:
            #     new_dist = distance(w, cluster)
            #     if (cluster.closest in (u,v)) and (new_dist > cluster.closest_dist):
            #         cluster.closest_dist = INF
            #         for other in self.heap:
            #             if other != cluster:
            #                 other_dist = distance(other, cluster)
            #                 if other_dist < cluster.closest_dist:
            #                     cluster.closest = other
            #                     cluster.closest_dist = other_dist
                
            #     if new_dist <= cluster.closest_dist:
            #         cluster.closest_dist = new_dist
            #         cluster.closest = w
            #     if new_dist < w.closest_dist:
            #         w.closest_dist = new_dist
            #         w.closest = cluster


            # calculate new distances from w to the others
            w.closest_dist = INF
            w.closest = None
            for cluster in self.heap:
                new_dist = distance(w, cluster)
                # if cluster's closest was one of the merged guys
                if cluster.closest == u or cluster.closest == v:
                    #print '!!!Closest was u or v', cluster.id
                    # if new merged is even closer or the same,
                    # no need to search for a new closest neighbor
                    if new_dist <= cluster.closest_dist:
                        #print '!!!New cluster closer than this guys old closest (u or v), SET as closest', cluster.id
                        cluster.closest_dist = new_dist
                        cluster.closest = w
                    # but if new merged guy is farther away, someone else
                    # COULD be closer -- so we'll check everyone, including
                    # the new guy
                    else:
                        #print '!!!This guy had u or v as closest, we will find you a new guy, bud', cluster.id
                        # delete the old dist, we'll find the new one.
                        cluster.closest_dist = INF
                        cluster.closest = None
                        for other in self.heap:
                            # skip over yourself (you can't be your own nearest neighbor)
                            if other != cluster:
                                other_dist = distance(other, cluster)
                                # if this guy is closer than the current nearest neighbor,
                                # it will kick the old guy out and take that place
                                if other_dist <= cluster.closest_dist:
                                    cluster.closest = other
                                    cluster.closest_dist = other_dist
                        # We didn't check if the new merged guy is even closer than all the others
                        # If it is, it will kick the current nearest neighbor and take its place
                        if new_dist <= cluster.closest_dist:
                            cluster.closest = w
                            cluster.closest_dist = new_dist
                # Now, if the cluster's closest was NOT one of the merged guys,
                # we only need to check if the new merged cluster is closer than
                # its current nearest neighbor, and update if necessary
                else:
                    #print "!!!This guy didn't even know u or v, but perhaps w is now even closer?", cluster.id
                    if new_dist <= cluster.closest_dist:
                        #print '!!!Yes it was!, hello, new neighbor.'
                        cluster.closest_dist = new_dist
                        cluster.closest = w

                # We updated everybody's closest neighbors except the new merged guy's
                # Check if this cluster is closest to w
                # If so, it is w's new nearest neighbor
                if new_dist < w.closest_dist:
                    #print '!!!This guy is closer than current dist, w has a new neighbor!', new_dist, w.closest_dist, cluster.id
                    w.closest_dist = new_dist
                    w.closest = cluster

            # insert w into the heap
            heapq.heappush(self.heap,w)

        # report on clusters
        # print '\nCLUSTERS\n------'
        # for cl in self.heap:
        #     print cl
        #     print '-------'

        clusters = []
        for point in inputVector:
            for clusterID, cluster in enumerate(self.heap):
                goToNextPoint = False
                for cluster_point in cluster.points:
                    if np.all(point == cluster_point):
                        clusters.append( clusterID )
                        goToNextPoint = True
                        break
                if goToNextPoint:
                    break

        return clusters
        




if __name__ == '__main__':
    


    data = np.array([[100,2,3],
                     [100,5,6],
                     [100,18,19],
                     [100,6,8],
                     [0, 300, 65],
                     [0, 340, 78],
                     [2346,2364,2360],
                     [2345,2354,2350],
                     [2342,2351,2355],
                     [2343,2359,2353],
                     [2344,2351,2351] ] )
    
    c = CURE()

    # Cluster with num_clusters = 3
    print c.cluster(data, 3)    #or: print c.cluster(data, 3, stop_criterion='num_clusters')


    # Cluster with distance threshold = 0.001
    print c.cluster(data, .001, stop_criterion='distance_threshold')




    
