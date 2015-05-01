
import numpy as np
import sys
from numpy.testing import assert_almost_equal
from KmeansParallel_func import KmeansParallel
from MP_func import KmeansParallel_mc

def test_level_label():
    for i in range(10):
        data = np.random.normal(size=(10,2))
        k = 3
        assert (len(set(KmeansParallel(n_clusters = k, data = data, l = 2*k).labels_)) == k) and (len(set(KmeansParallel_mc(n_clusters = k, data = data, l = 2*k).labels_)) == k)

def test_num_label():
    for i in range(10):
        data = np.random.normal(size=(10,2))
        k = 3
        assert (len(KmeansParallel(n_clusters = k, data = data, l = 2*k).labels_) == len(data)) and (len(KmeansParallel_mc(n_clusters = k, data = data, l = 2*k).labels_) == len(data))

        