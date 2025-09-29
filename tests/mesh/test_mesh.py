import pytest
from flatiron_tk.mesh import Mesh
import numpy as np


import dolfinx.mesh 
from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()
MPI = import_mpi4py()

def test_line_mesh(line_mesh_1, vector_equal):
    '''
    Test the creation of a 1D line mesh.
    '''
    dolfin_mesh = line_mesh_1
    subdomain = dolfinx.mesh.meshtags(dolfin_mesh, dolfin_mesh.topology.dim, np.array([]), np.array([]))
    boundary = dolfinx.mesh.meshtags(dolfin_mesh, dolfin_mesh.topology.dim-1, np.array([]), np.array([]))

    mesh = Mesh(mesh=dolfin_mesh)
    coordinates = mesh.msh.geometry.x # always 3D 

    lower_bnd = np.min(coordinates, axis=0)
    upper_bnd = np.max(coordinates, axis=0)

    assert vector_equal(lower_bnd, np.array([0.0, 0.0, 0.0])), 'Lower bound should be (0.0, 0.0)'
    assert vector_equal(upper_bnd, np.array([1.0, 0.0, 0.0])), 'Upper bound should be (1.0, 1.0)'
    assert mesh.msh is not None, 'Mesh should be initialized.'
    assert mesh.get_tdim() == 1, 'Topological dimension should be 1.'
    assert mesh.get_fdim() == 0, 'Facet dimension should be 0.'
    assert mesh.get_gdim() == 1, 'Geometric dimension should be 1.'
    assert mesh.boundary is not None, 'Boundary should be initialized.'
    assert mesh.subdomain is not None, 'Subdomain should be initialized.'

def test_rect_mesh(box_mesh_2, vector_equal):
    '''
    Test the creation of a 2D rectangular mesh.
    '''
    dolfin_mesh = box_mesh_2
    subdomain = dolfinx.mesh.meshtags(dolfin_mesh, dolfin_mesh.topology.dim, np.array([]), np.array([]))
    boundary = dolfinx.mesh.meshtags(dolfin_mesh, dolfin_mesh.topology.dim-1, np.array([]), np.array([]))

    mesh = Mesh(mesh=dolfin_mesh)
    coordinates = mesh.msh.geometry.x # always 3D 

    lower_bnd = np.min(coordinates, axis=0)
    upper_bnd = np.max(coordinates, axis=0)

    assert vector_equal(lower_bnd, np.array([0.0, 0.0, 0.0])), 'Lower bound should be (0.0, 0.0)'
    assert vector_equal(upper_bnd, np.array([1.0, 1.0, 0.0])), 'Upper bound should be (1.0, 1.0)'
    assert mesh.msh is not None, 'Mesh should be initialized.'
    assert mesh.get_tdim() == 2, 'Topological dimension should be 2.'
    assert mesh.get_fdim() == 1, 'Facet dimension should be 1.'
    assert mesh.get_gdim() == 2, 'Geometric dimension should be 2.'
    assert mesh.boundary is not None, 'Boundary should be initialized.'
    assert mesh.subdomain is not None, 'Subdomain should be initialized.'
    
def test_box_mesh(box_mesh_3, vector_equal):
    '''
    Test the creation of a 3D box mesh.
    '''
    dolfin_mesh = box_mesh_3
    subdomain = dolfinx.mesh.meshtags(dolfin_mesh, dolfin_mesh.topology.dim, np.array([]), np.array([]))
    boundary = dolfinx.mesh.meshtags(dolfin_mesh, dolfin_mesh.topology.dim-1, np.array([]), np.array([]))

    mesh = Mesh(mesh=dolfin_mesh)
    coordinates = mesh.msh.geometry.x # always 3D 

    lower_bnd = np.min(coordinates, axis=0)
    upper_bnd = np.max(coordinates, axis=0)

    assert vector_equal(lower_bnd, np.array([0.0, 0.0, 0.0])), 'Lower bound should be (0.0, 0.0)'
    assert vector_equal(upper_bnd, np.array([1.0, 1.0, 1.0])), 'Upper bound should be (1.0, 1.0)'
    assert mesh.msh is not None, 'Mesh should be initialized.'
    assert mesh.get_tdim() == 3, 'Topological dimension should be 3.'
    assert mesh.get_fdim() == 2, 'Facet dimension should be 2.'
    assert mesh.get_gdim() == 3, 'Geometric dimension should be 3.'
    assert mesh.boundary is not None, 'Boundary should be initialized.'
    assert mesh.subdomain is not None, 'Subdomain should be initialized.'

def test_msh_file():
    '''
    Test reading a mesh from a GMSH file.
    '''
    mesh_file = 'mesh/test_meshes/box.msh'
    mesh = Mesh(mesh_file=mesh_file)

    assert mesh.get_gdim() == 2, 'Geometric dimension should be 2.'
    assert mesh.get_tdim() == 2, 'Topological dimension should be 2.'
    assert mesh.get_fdim() == 1, 'Facet dimension should be 1.'

@pytest.fixture
def ubm_marked_bnd(ubm_flatiron, float_equal):
    """
    Create a unit box mesh with a marked boundary.
    """
    mesh = ubm_flatiron
    mesh.mark_boundary({1: lambda x: np.isclose(x[0], 0)})
    return mesh

def test_mark_boundary(ubm_marked_bnd, float_equal):
    """
    Test boundary marking function.
    """
    boundary_facets = ubm_marked_bnd.boundary.find(1)
    facet_midpoints = []

    for facet in boundary_facets:
        facet_vertices = ubm_marked_bnd.msh.topology.connectivity(ubm_marked_bnd.get_fdim(), 0).links(facet)
        facet_coords = ubm_marked_bnd.msh.geometry.x[facet_vertices]
        facet_midpoints.append(np.mean(facet_coords, axis=0))

    for midpoint in facet_midpoints:
        assert float_equal(midpoint[0], 0)

def test_mean_boundary_normal(ubm_marked_bnd, vector_equal):
    """
    Test mean boundary of the first y-side of a Unit Box . 
    """
    mean_bnd_nrm = ubm_marked_bnd.get_mean_boundary_normal(1)
    assert vector_equal(mean_bnd_nrm, [-1, 0, 0]), "Mean boundary normal incorrect."

def test_mean_cell_diameter(ubm_marked_bnd, float_equal):
    """
    Test the average cell diameter of a unit box mesh split 
    into cubes of 1/10x1/10x1/10 is equal to the length of 
    the sub-cube's diagonal cross section.
    """
    mean_cell_dmtr = ubm_marked_bnd.get_mean_cell_diameter()
    print(mean_cell_dmtr)
    assert float_equal(mean_cell_dmtr, np.sqrt(3)/10), "Average cell diameter incorrect."