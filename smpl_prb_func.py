
import numpy as np
from dist_sq_func import dist_sq
from cost_func import cost

#sample probability function
def smpl_prb(c,data,l):
    phi_temp = cost(c,data)
    return np.array([(min(dist_sq(c, d, axis = 1)))*l/phi_temp for d in data])