import numpy as np
import pytest
from flatiron_tk.mesh import Mesh, LineMesh, RectMesh, BoxMesh

def approx_zero(a):
    '''
    Return whether a is aproximately 0
    '''
    return a == pytest.approx(0.0)

def squared_distance(a, b):
    return np.dot(a-b, a-b)

@pytest.fixture(scope="module")
def float_equal():

    def _float_equal(a, b):
        return approx_zero(abs(a-b))

    return _float_equal

@pytest.fixture(scope="module")
def vector_equal():

    def _vector_equal(a, b):
        return approx_zero(squared_distance(np.array(a), np.array(b)))
    
    return _vector_equal

@pytest.fixture(scope="module")
def unit_box_mesh_fe():
    import fenics as fe
    return fe.BoxMesh(fe.Point(0, 0, 0), fe.Point(1,1,1), 10, 10, 10)

@pytest.fixture(scope="module")
def ubm_flatiron(unit_box_mesh_fe):
    return Mesh(mesh=unit_box_mesh_fe)

@pytest.fixture(scope="module")
def mesh_3d():
    return BoxMesh(0, 0, 0, 1, 1, 1, 1/10)

@pytest.fixture(scope="module")
def mesh_2d():
    return RectMesh(0, 0, 1, 1, 1/10)

@pytest.fixture(scope="module")
def mesh_1d():
    return LineMesh(0, 1, 1/10)
