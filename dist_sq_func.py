
import numpy as np

#distance square function
def dist_sq(a, b, axis = 0):
    return np.sum((a-b)**2,axis)