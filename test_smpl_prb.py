
import numpy as np
from numpy.testing import assert_almost_equal
from cost_func import cost
from smpl_prb_func import smpl_prb
from MP_func import smpl_prb_mc

def test_non_negativity():
    l = 3
    for i in range(10):
        data = np.random.normal(size=(10,2))
        c = data[np.random.choice(range(10),1),]
        assert np.alltrue(smpl_prb(c,data,l) >= 0) and np.alltrue(smpl_prb_mc(c,data,l,axis=1) >= 0)
        

def test_sum_to_l():
    for i in range(10):
        l = i + 1
        data = np.random.normal(size=(10,2))
        c = data[np.random.choice(range(10),1),]
        assert ((np.sum(smpl_prb(c,data,l)) - l) <= 1e-6) and ((np.sum(smpl_prb_mc(c,data,l,axis=1)) - l) <= 1e-6)

def test_in_c_zero():
    l = 2
    for i in range(10):
        data = np.random.normal(size=(10,2))
        choice = np.random.choice(range(10),1)
        c = data[choice,]
        prb = smpl_prb(c,data,l)
        prb_mc = smpl_prb_mc(c,data,l,axis=1)
        assert np.alltrue(prb[choice,] == 0) and np.alltrue(prb_mc[choice,] == 0)
