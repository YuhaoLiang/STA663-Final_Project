
import numpy as np
from numpy.testing import assert_almost_equal
from dist_sq_func import dist_sq

def test_non_negativity():
    for i in range(10):
        u = np.random.normal(3)
        v = np.random.normal(3)
        assert dist_sq(u, v) >= 0

def test_coincidence_when_zero():
    u = np.zeros(3)
    v = np.zeros(3)
    assert dist_sq(u, v) == 0

def test_coincidence_when_not_zero():
     for i in range(10):
        u = np.random.random(3)
        v = np.zeros(3)
        assert dist_sq(u, v) != 0

def test_symmetry():
    for i in range(10):
        u = np.random.random(3)
        v = np.random.random(3)
        assert dist_sq(u, v) == dist_sq(v, u)

def test_triangle():
    
    u = np.random.random(3)
    v = np.random.random(3)
    w = np.random.random(3)
    assert np.sqrt(dist_sq(u, w)) <= np.sqrt(dist_sq(u, v)) + np.sqrt(dist_sq(v, w))

def test_known1():
    u = np.array([0])
    v = np.array([3])
    assert_almost_equal(dist_sq(u, v), 3**2)

def test_known2():
    u = np.array([0,0])
    v = np.array([3, 4])
    assert_almost_equal(dist_sq(u, v), 5**2)

def test_known3():
    u = np.array([0,0])
    v = np.array([-3, -4])
    assert_almost_equal(dist_sq(u, v), 5**2)