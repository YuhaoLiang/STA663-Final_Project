from __future__ import division
import os
import numpy as np
import pandas as pd
import sklearn.cluster
# Kmeans++ function just for the convenience of benchmark and comparison of efficiency
def KmeansPlus(n_clusters_, data, n_init_=10, max_iter_=300, tol_=0.0001):
    import sklearn.cluster
    km1 = sklearn.cluster.KMeans(n_clusters = n_clusters_, init='k-means++', n_init=n_init_, max_iter=max_iter_, tol=tol_);
    km1.fit(data);
    return km1