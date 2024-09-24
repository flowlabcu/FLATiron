import fenics as fe
from .mesh import Mesh

class bnd(fe.SubDomain):
    '''
    Define the boundary of a cartesian mesh such that x[d] == x_bnd
    by defining the dimension and the point location at the
    boundary
    '''
    def __init__(self, x_bnd, d, *args, **kwargs):
        self.x_bnd = x_bnd
        self.d = d
        super().__init__(*args, **kwargs)

    def inside(self, x, on_boundary):
        # return on_boundary and fe.near( x[self.d], self.x_bnd )
        return on_boundary and abs(x[self.d]-self.x_bnd) < 1e-8

def _cart_mesh(comm, x0, x1, dx, *args):
    '''
    General function to create a cartesian mesh
    bounded by point x0 and point x1
    '''
    # Generate number of elements in each dimension
    assert( len(x0)==len(x1)==len(dx) )
    dim = len(x0)
    ne = [ int( (x1[i]-x0[i])/dx[i] ) for i in range(dim) ]

    # Generate mesh
    if dim == 1:
        mesh = fe.IntervalMesh(comm, ne[0], x0[0], x1[0], *args)
    elif dim == 2:
        mesh = fe.RectangleMesh(comm, fe.Point(x0), fe.Point(x1), *ne , *args)
    else:
        mesh = fe.BoxMesh(comm, fe.Point(x0), fe.Point(x1), *ne , *args)

    # Mark boundaries
    boundary = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary.set_all(0)
    k = 1
    for i,xi in enumerate(x0):
        bnd(xi, i).mark(boundary, k)
        k+=1
    for i,xi in enumerate(x1):
        bnd(xi, i).mark(boundary, k)
        k+=1

    return mesh, boundary

class LineMesh(Mesh):
    """
    (x0) ------------------- (x1)
    """

    def __init__(x0, x1, dx, comm=fe.MPI.comm_world):
        mesh, boundary = _cart_mesh(comm, [x0], [x1], [dx])
        super().__init__(mesh=mesh, boundary=boundary)

class RectMesh(Mesh):
    """
              (x1,y1)
     ___________
    |          |
    |          |
    |__________|
  (x0,y0)
    """
    def __init__(self, x0, y0, x1, y1, dx, comm=fe.MPI.comm_world, *args):
        if isinstance(dx, int) or isinstance(dx, float):
            dx_lst = [dx for i in range(2)]
        else:
            dx_lst = dx
        mesh, boundary = _cart_mesh(comm, [x0,y0], [x1,y1], dx_lst, *args)
        super().__init__(mesh=mesh, boundary=boundary)

class BoxMesh(Mesh):
    def __init__(self, x0, y0, z0, x1, y1, z1, dx, comm=fe.MPI.comm_world, *args):

        """
                           (x1,y1,z1)
                    __________
                   /|        /
                  / |       /|
                 /  |______/_|
                /___/_____/  /
                |  /     |  /
                | /      | /
                |/_______|/
            (x0,y0,z0)

        """



        if isinstance(dx, int) or isinstance(dx, float):
            dx_lst = [dx for i in range(3)]
        else:
            dx_lst = dx
        mesh, boundary = _cart_mesh(comm, [x0,y0,z0], [x1,y1,z1], dx_lst, *args)
        super().__init__(mesh=mesh, boundary=boundary)

