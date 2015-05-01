from __future__ import division
import os
import sys
import glob
from tabulate import tabulate
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
plt.style.use('ggplot')
df = pd.read_csv('SimData.csv')
data = np.array(df)
#number of cluster
k = 3

with open('table.tex', 'w') as f:
    f.write(tabulate(df.ix[:10,], headers=list(df.columns),tablefmt="latex", floatfmt=".4f"))

from KMeansPlusPlus import KmeansPlus
km1 = KmeansPlus(n_clusters_=k, data=data, n_init_=10, max_iter_=300, tol_=0.0001);
plt.figure(tight_layout=True);
plt.scatter(df.X,df.Y,c=km1.labels_);
plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],marker='o',s=50);
plt.title("k-means++ cluster");
plt.savefig('K-meansPlusPlus.png');

from KMeansParallel import KmeansParallel
km = KmeansParallel(n_clusters = k, data = data, l = 2*k);
# plt.figure(figsize=(12, 9));
plt.figure(tight_layout=True);
plt.scatter(df.X,df.Y,c=km.labels_);
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker='o',s=50);
plt.title("k-means|| cluster");
plt.savefig('K-meansParallel.png');

from KMeansParallel_MC import KmeansParallel_mc
km_mc = KmeansParallel_mc(n_clusters = k, data = data, l = 2*k);
# plt.figure(figsize=(12, 9));
plt.figure(tight_layout=True);
plt.scatter(df.X,df.Y,c=km_mc.labels_);
plt.scatter(km_mc.cluster_centers_[:,0],km_mc.cluster_centers_[:,1],marker='o',s=50);
plt.title("Multiple cores version k-means|| cluster");
plt.savefig('K-meansParallel_MC.png');