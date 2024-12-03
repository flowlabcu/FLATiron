import pytest
import numpy as np
import sys
from flatiron_tk.functions import profiles

def test_plug(vector_equal):
    U = 10
    prof = profiles.plug(0, [1,0,0], U)
    assert vector_equal(prof, [10, 0, 0])

def test_parabolic_2D(vector_equal):
    xc = np.array([0, 0.5])
    n = [1, 0]
    U = 10
    r = 0.3
    u = []
    assert vector_equal(profiles.parabolic_2d(xc, n, U, xc, r)[0], U)




