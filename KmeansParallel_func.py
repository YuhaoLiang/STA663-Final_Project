
from __future__ import division
import numpy as np
import sys
import sklearn.cluster
from dist_sq_func import dist_sq
from cost_func import cost
from smpl_prb_func import smpl_prb
from weight_func_file import weight_func

#Kmeans||
def KmeansParallel(n_clusters, data, l):
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
    phi = cost(c,data)
    
    #Step 3~6 - get potential centers
    for i in range(np.ceil(np.log(phi)).astype(int)):
        c_add = data[smpl_prb(c,data,l)>np.random.uniform(size = num),]
        c = np.concatenate((c,c_add))
        
    #Step 7
    # Find the closet point in c for each point in data
    ##weight
    weight = weight_func(c, data)
    
    #Step 8 - recluster by kmeans++ initialization
    c_final = data[np.random.choice(range(len(c)),size=1,p=weight),]
    data_final = c
    for i in range(n_clusters-1):
        new_prb = smpl_prb(c_final,data_final,l) * weight
        c_fin_add = data[np.random.choice(range(len(c)),size=1,p=new_prb/np.sum(new_prb)),]
        c_final = np.concatenate((c_final,c_fin_add))
    
    #Do k-means with initial centers
    import sklearn.cluster
    km2 = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init=1, init=c_final, max_iter=500, tol=0.0001)
    km2.fit(data);
    
    #return a KMeans type result - including: cluster_centers_, labels_, inertia_
    return km2
    