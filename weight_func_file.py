
import numpy as np
from dist_sq_func import dist_sq

#weight function - propotional to the number of data points have the same specific center
def weight_func(c, data):
    # Find the closet point in c for each point in data
    min_c = [np.argmin(dist_sq(c, d, axis = 1)) for d in data];
    ## number of points which is closest to each s in c
    num_closest = np.array([min_c.count(i) for i in range(len(c))]).astype(float);
    ## return normalized weight
    return num_closest/np.sum(num_closest)