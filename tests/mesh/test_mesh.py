import pytest
from flatiron_tk.mesh import Mesh
import numpy as np

# ------------------------------------------------------- #


@pytest.fixture

# Function def returns mesh? yes
def ubm_marked_bnd(ubm_flatiron, float_equal):
    def on_x0_face(x): 
        return float_equal(x[0], 0)
    mesh = ubm_flatiron
    mesh.mark_boundary(1, on_x0_face)
    return mesh

# ubm_marked_bnd input as parameter?
def test_mark_boundary(ubm_marked_bnd, float_equal):
    for i in np.where(ubm_marked_bnd.boundary.array()==1)[0]:
        assert float_equal(ubm_marked_bnd.facet(i).midpoint()[0], 0)

def test_flat_boundary_normal(ubm_marked_bnd, vector_equal):
    #shouldnt 
    n = ubm_marked_bnd.flat_boundary_normal(1)
    assert vector_equal(n, [1, 0, 0]) # This function returns invert normal

def test_mean_boundary_normal(ubm_marked_bnd, vector_equal):
    '''
    Mean boundary normal of the first y-side of a Unit *Box* 
    Mesh points only in the negative x-direction.
    '''
    mean_bnd_nrm = ubm_marked_bnd.mean_boundary_normal(1)
    assert vector_equal(mean_bnd_nrm, [-1, 0, 0]), "Mean boundary normal incorrect."


def test_avg_cell_diameter(ubm_marked_bnd, float_equal):
    '''
    Avg cell diameter of a Unit *Box* Mesh split into
    cubes of 1/10x1/10x1/10 is equal to the length of 
    the sub-cubes's diagonal cross section.
    '''
    avg_cell_dmtr = ubm_marked_bnd.avg_cell_diameter()
    assert float_equal(avg_cell_dmtr, np.sqrt(3)/10), "Average cell diameter incorrect."