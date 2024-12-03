import pytest
from flatiron_tk.mesh import LineMesh, RectMesh, BoxMesh
import numpy as np

@pytest.fixture
def dx():
    return 1e-1



# ---------------------------------------------------------

def test_line_mesh(dx, float_equal):
    x0 = np.random.uniform(0, 1)
    x1 = np.random.uniform(2, 3)
    mesh = LineMesh(x0, x1, dx)
    mx = mesh.fenics_mesh().coordinates()
    bnda = mesh.boundary.array()
    assert float_equal( np.min(mx), x0 )
    assert float_equal( np.max(mx), x1 )

    left_bnd = np.where(bnda==1)[0]
    right_bnd = np.where(bnda==2)[0]
    for i in left_bnd:
        facet = mesh.facet(i)
        assert float_equal( facet.midpoint()[0], x0 )
    for i in right_bnd:
        facet = mesh.facet(i)
        assert float_equal( facet.midpoint()[0], x1 )

def test_rect_mesh(dx, float_equal, vector_equal):

    x0 = np.random.uniform(0, 1)
    y0 = np.random.uniform(0, 1)

    x1 = np.random.uniform(2, 3)
    y1 = np.random.uniform(2, 3)

    mesh = RectMesh(x0, y0, x1, y1, dx)
    mx = mesh.fenics_mesh().coordinates()
    bnda = mesh.boundary.array()
    lower_bnd = np.min(mx, axis=0)
    upper_bnd = np.max(mx, axis=0)
    
    assert vector_equal( lower_bnd, [x0, y0] )
    assert vector_equal( upper_bnd, [x1, y1] )

    left_bnd = np.where(bnda==1)[0]
    bot_bnd = np.where(bnda==2)[0]
    right_bnd = np.where(bnda==3)[0]
    top_bnd = np.where(bnda==4)[0]

    for i in left_bnd:
        assert float_equal( mesh.facet(i).midpoint()[0], x0 )

    for i in bot_bnd:
        assert float_equal( mesh.facet(i).midpoint()[1], y0 )

    for i in right_bnd:
        assert float_equal( mesh.facet(i).midpoint()[0], x1 )

    for i in top_bnd:
        assert float_equal( mesh.facet(i).midpoint()[1], y1 )

def test_box_mesh(dx, float_equal, vector_equal):

    x0 = np.random.uniform(0, 1)
    y0 = np.random.uniform(0, 1)
    z0 = np.random.uniform(0, 1)

    x1 = np.random.uniform(2, 3)
    y1 = np.random.uniform(2, 3)
    z1 = np.random.uniform(2, 3)

    mesh = BoxMesh(x0, y0, z0, x1, y1, z1, dx)
    mx = mesh.fenics_mesh().coordinates()
    bnda = mesh.boundary.array()
    lower_bnd = np.min(mx, axis=0)
    upper_bnd = np.max(mx, axis=0)
    
    assert vector_equal( lower_bnd, [x0, y0, z0] )
    assert vector_equal( upper_bnd, [x1, y1, z1] )

    x0_bnd = np.where(bnda==1)[0]
    y0_bnd = np.where(bnda==2)[0]
    z0_bnd = np.where(bnda==3)[0]
    x1_bnd = np.where(bnda==4)[0]
    y1_bnd = np.where(bnda==5)[0]
    z1_bnd = np.where(bnda==6)[0]

    for i in x0_bnd:
        assert float_equal( mesh.facet(i).midpoint()[0], x0 )

    for i in y0_bnd:
        assert float_equal( mesh.facet(i).midpoint()[1], y0 )

    for i in z0_bnd:
        assert float_equal( mesh.facet(i).midpoint()[2], z0 )

    for i in x1_bnd:
        assert float_equal( mesh.facet(i).midpoint()[0], x1 )

    for i in y1_bnd:
        assert float_equal( mesh.facet(i).midpoint()[1], y1 )

    for i in z1_bnd:
        assert float_equal( mesh.facet(i).midpoint()[2], z1 )
        

