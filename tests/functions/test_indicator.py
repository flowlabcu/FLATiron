import pytest
import numpy as np
from flatiron_tk.functions import IndicatorFieldScalar
import fenics as fe

def test_indicator(ubm_flatiron, float_equal):
    V = fe.FunctionSpace(ubm_flatiron.fenics_mesh(), 'CG', 1)
    r = 0.2
    xc = np.array([0.5, 0.5, 0.5])
    def domain(x):
        return np.dot(x-xc, x-xc) < r**2
    I = IndicatorFieldScalar(domain)
    I = fe.interpolate(I, V)
    assert float_equal( I(fe.Point(xc)), 1.0 )
    assert float_equal( I(fe.Point([0.6,0.6,0.6])), 1.0 )
    assert float_equal( I(fe.Point([0, 0, 0])), 0.0 )

