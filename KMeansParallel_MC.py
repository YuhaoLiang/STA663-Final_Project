from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from multiprocessing import Pool, cpu_count
from functools import partial

##distance square function - we don't need the square root so we can save computation time
# euclidean distance
def dist_sq(a, b, axis = 0):
    return np.sum((a-b)**2,axis)

##minimum distance square between data and centers
def min_dist_sq(d, c, axis):
    return np.min(dist_sq(c,d,axis))

##cost function
# Version 4 - parallel computing with multiple cores
def cost_mc(c,data,axis=1):
    pool = Pool(processes=cpu_count())
    # define a partial function for min_dist_sq
    partial_min_dist_sq = partial(min_dist_sq, c=c, axis=1)
    cost = np.sum(pool.map(partial_min_dist_sq, data))
    pool.close()
    pool.terminate()
    return cost


#sampling probability function
# Version 4 - parallel computing with multiple cores
def smpl_prb_mc(c,data,l,axis):
    phi_temp = cost_mc(c,data,axis)
    pool = Pool(processes=cpu_count())
    # define a partial function for min_dist_sq
    partial_min_dist_sq2 = partial(min_dist_sq, c=c, axis=1)
    sampling_prob = np.array(pool.map(partial_min_dist_sq2, data))*l/phi_temp
    pool.close()
    pool.terminate()
    return sampling_prob

##weight function 
# propotional to the number of data points have the same specific center
# Version 2
def weight_func(c, data):
    # Find the closet point in c for each point in data
    min_c = [np.argmin(dist_sq(c, d, axis = 1)) for d in data];
    ## number of points which is closest to each s in c
    num_closest = np.array([min_c.count(i) for i in range(len(c))]).astype(float);
    ## return normalized weight
    return num_closest/np.sum(num_closest)


#Kmeans||
#l is oversampling factor

def KmeansParallel_mc(n_clusters, data, l):
    if n_clusters <= 0 or not(isinstance(n_clusters,int)):
        sys.exit("n_cluster is not positive integer")
    
    if l <= 0: 
        sys.exit("l is not positive")
    
    if len(data) < n_clusters: 
        sys.exit("number of data is less than n_clusters")
    
    num = len(data)
    
    #Step 1 - uniformly sample one point
    c = np.array(data[np.random.choice(range(num),1),])
    
    #Step 2 - cost
    phi = cost_mc(c,data,axis=1)
    
    #Step 3~6 - get potential centers
    for i in range(np.ceil(np.log(phi)).astype(int)):
        c_add = data[smpl_prb_mc(c,data,l,axis=1)>np.random.uniform(size = num),]
        c = np.concatenate((c,c_add))
        
    #Step 7
    # Find the closet point in c for each point in data
    ##weight
    weight = weight_func(c,data)
    
    #Step 8 - recluster by kmeans++ initialization
    c_final = data[np.random.choice(range(len(c)),size=1,p=weight),]
    data_final = c
    for i in range(n_clusters-1):
        new_prb = smpl_prb_mc(c_final,data_final,l,axis=1) * weight
        c_fin_add = data[np.random.choice(range(len(c)),size=1,p=new_prb/np.sum(new_prb)),]
        c_final = np.concatenate((c_final,c_fin_add))
    
    
    #Do k-means with initial centers
    import sklearn.cluster
    km2 = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init=1, init=c_final, max_iter=500, tol=0.0001)
    km2.fit(data);
    
    #return a KMeans type result - including: cluster_centers_, labels_, inertia_
    return km2
    