import numpy as np
from flatiron_tk.mesh import Mesh
import pytest

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()
MPI = import_mpi4py()

from mpi4py import MPI

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
def line_mesh_1():
    """Create a 1D line mesh for testing."""
    # Define the line length
    length = 1.0
    
    # Create a line mesh using dolfinx
    return dolfinx.mesh.create_interval(comm = MPI.COMM_WORLD, 
                                     points = (0.0, length),
                                     nx = 32)

@pytest.fixture(scope="module")
def box_mesh_2():
    """Create a 2D box mesh for testing."""
    # Define the box dimensions
    length = 1.0
    width = 1.0
    
    # Create a box mesh using dolfinx
    return dolfinx.mesh.create_rectangle(comm = MPI.COMM_WORLD, 
                                         points = ((0.0, 0.0), (length, width)),
                                         n = (32, 32),
                                         cell_type = dolfinx.mesh.CellType.triangle)

@pytest.fixture(scope="module")
def box_mesh_3():
    """Create a 3D box mesh for testing."""
    # Define the box dimensions
    length = 1.0
    width = 1.0
    height = 1.0
    
    # Create a box mesh using dolfinx
    return dolfinx.mesh.create_box(comm = MPI.COMM_WORLD, 
                                         points = ((0.0, 0.0, 0.0), (length, width, height)),
                                         n = (32, 32, 32),
                                         cell_type = dolfinx.mesh.CellType.tetrahedron)

@pytest.fixture(scope="module")
def unit_box_mesh():
    """
    Dolfin mesh.
    """
    return dolfinx.mesh.create_box(comm = MPI.COMM_WORLD, 
                                   points = ((0, 0, 0), (1, 1, 1)), 
                                   n = (10, 10, 10),
                                   cell_type = dolfinx.mesh.CellType.hexahedron)

@pytest.fixture(scope="module")
def ubm_flatiron(unit_box_mesh):
    """
    Create a flatiron mesh object.
    """
    return Mesh(mesh=unit_box_mesh)
