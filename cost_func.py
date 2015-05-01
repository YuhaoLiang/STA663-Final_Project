
import numpy as np
from dist_sq_func import dist_sq

##cost function
def cost(c,data):
    return np.sum([min(dist_sq(c, d, axis = 1)) for d in data])