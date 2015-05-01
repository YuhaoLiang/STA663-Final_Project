
import numpy as np
from numpy.testing import assert_almost_equal
from cost_func import cost
from MP_func import cost_mc

def test_non_negativity():
    for i in range(10):
        data = np.random.normal(size=(10,2))
        c = data[np.random.choice(range(10),1),]
        assert (cost(c, data) >= 0) and (cost_mc(c, data) >= 0)

def test_full_data_zero():
    for i in range(10):
        data = np.random.normal(size=(10,2))
        c = data
        assert (cost(c, data) == 0) and (cost_mc(c, data) == 0)

def test_c_more_cost_less():
     for i in range(10):
        data = np.random.normal(size=(10,2))
        c_more = data[np.random.choice(range(10),4,replace=False),]
        c = c_more[:2,]
        assert (cost(c_more, data) <= cost(c, data)) and (cost_mc(c_more, data) <= cost_mc(c, data))