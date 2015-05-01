
import numpy as np
from numpy.testing import assert_almost_equal
from cost_func import cost
from weight_func_file import weight_func

def test_non_negativity():
    for i in range(10):
        data = np.random.normal(size=(10,2))
        c = data[np.random.choice(range(10),1),]
        assert np.alltrue(weight_func(c,data) >= 0)
        

def test_sum_to_1():
    for i in range(10):
        data = np.random.normal(size=(10,2))
        c = data[np.random.choice(range(10),1),]
        assert (np.sum(weight_func(c,data)) - 1) <= 1e-6