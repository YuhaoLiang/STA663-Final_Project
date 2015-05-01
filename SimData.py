import os
import numpy as np
import pandas as pd
import random
random.seed(42)
#Data simulation - mixture of 3 binary normal distribution
##distribution mean & covariance
mean1 = np.array([3,5])
mean2 = np.array([-2,3])
mean3 = np.array([-6,-1])
cov1 = np.array([[1,0],[0,2]])
cov2 = np.array([[1,-0.6],[-0.6,1]])
cov3 = np.array([[3,0.3],[0.3,1]])

n = 2000
N = 3*n
data = np.vstack((np.random.multivariate_normal(mean1, cov1,n),np.random.multivariate_normal(mean2, cov2,n),
                  np.random.multivariate_normal(mean3, cov3,n)))
data = data[np.random.choice(range(N),size = N, replace=False),]
df = pd.DataFrame(data,columns=["X","Y"])
df.index = pd.Index(range(N))

#Save simluated data as csv file
df.to_csv('SimData.csv', index=False)