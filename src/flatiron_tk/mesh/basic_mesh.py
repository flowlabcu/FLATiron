import numpy as np

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()
MPI = import_mpi4py()

from .mesh import Mesh

def _cartesian_mesh(x0, x1, dx, comm, **kwargs):
    """
    Parameters
    ----------
    x0 : array-like
        Start point of the mesh (list or array of coordinates).
    x1 : array-like
        End point of the mesh (list or array of coordinates).
    dx : array-like
        Cell diameter in each dimension.
    comm : MPI.Comm
        MPI communicator, typically ``MPI.COMM_WORLD``.
    cell_type : dolfinx.mesh.CellType, optional
        Dolfinx cell type (e.g., triangle, tetrahedron). If not provided, defaults to triangle for 2D and tetrahedron for 3D.
    Returns
    -------
    msh : dolfinx.mesh.Mesh
        The created dolfinx mesh object.
    Raises
    ------
    AssertionError
        If ``x0``, ``x1``, and ``dx`` do not have the same length.
    Notes
    -----
    - For 1D, creates an interval mesh.
    - For 2D, creates a rectangle mesh (default cell type: triangle).
    - For 3D, creates a box mesh (default cell type: tetrahedron).
    """
    
    assert(len(x0) == len(x1)== len(dx)), "x0, x1, and dx must have the same length"
    
    dim = len(x0)
    num_elements = [int((x1[i] - x0[i]) / dx[i]) for i in range(dim)]
    cell_type = kwargs.pop('cell_type', None) 

    if dim == 1:
        msh = dolfinx.mesh.create_interval(comm, num_elements[0], np.array([x0[0], x1[0]]))
    elif dim == 2:
        if cell_type is None:
            cell_type = dolfinx.mesh.CellType.triangle
        msh = dolfinx.mesh.create_rectangle(comm, [x0, x1], num_elements, cell_type)
    elif dim == 3:
        if cell_type is None:
            cell_type = dolfinx.mesh.CellType.tetrahedron
        msh = dolfinx.mesh.create_box(comm, [x0, x1], n=num_elements, cell_type=cell_type)
    
    return msh

class CuboidMesh(Mesh):
    """
    Create a 3D cuboid mesh between (x0, y0, z0) and (x1, y1, z1) with a given element size.
    Parameters:
    -------------
        x0: Start point of the mesh in the x-direction.
        y0: Start point of the mesh in the y-direction.
        z0: Start point of the mesh in the z-direction.
        x1: End point of the mesh in the x-direction.
        y1: End point of the mesh in the y-direction.
        z1: End point of the mesh in the z-direction.
        dx: Element size in each direction (can be a single float or a list/tuple of three floats).
        comm: MPI communicator, default is MPI.COMM_WORLD.
        **kwargs: Additional keyword arguments passed to dolfinx.mesh.create_box (e.g., cell_type).         
    """
    def __init__(self, x0, y0, z0, x1, y1, z1, dx, comm=MPI.COMM_WORLD, **kwargs):
        
        self.x0 = x0
        self.x1 = x1

        self.dx = dx
        self.comm = comm

        # Create the mesh
        dx_lst = [dx for i in range(3)]
        self.msh = _cartesian_mesh([x0, y0, z0], [x1,y1,z1], dx_lst, comm, **kwargs)

        markings = {
            1: lambda x: np.isclose(x[0], x0), 
            2: lambda x: np.isclose(x[1], y0),
            3: lambda x: np.isclose(x[2], z0),
            4: lambda x: np.isclose(x[0], x1),
            5: lambda x: np.isclose(x[1], y1),
            6: lambda x: np.isclose(x[2], z1)
        }

        self.mark_boundary(markings)

class RectMesh(Mesh):
    """
    Create a 2D rectangular mesh between (x0, y0) and (x1, y1) with a given element size.
    Parameters:
    -------------
        x0: Start point of the mesh in the x-direction.
        y0: Start point of the mesh in the y-direction.
        x1: End point of the mesh in the x-direction.
        y1: End point of the mesh in the y-direction.
        dx: Element size in each direction (can be a single float or a list/tuple of two floats).
        comm: MPI communicator, default is MPI.COMM_WORLD.
        **kwargs: Additional keyword arguments passed to dolfinx.mesh.create_rectangle (e.g., cell_type).
    """
    def __init__(self, x0, y0, x1, y1, dx, comm=MPI.COMM_WORLD, **kwargs):
        self.comm = comm

        # Create the mesh
        if isinstance(dx, (list, tuple, np.ndarray)):
            dx_lst = list(dx)
        else:
            dx_lst = [dx for _ in range(2)]

        if dx_lst.__len__() != 2:
            raise ValueError("dx_lst must be a list of length 2")

        self.msh = _cartesian_mesh([x0, y0], [x1, y1], dx_lst, comm,  **kwargs)

        markings = {
            1: lambda x: np.isclose(x[0], x0), 
            2: lambda x: np.isclose(x[1], y0),
            3: lambda x: np.isclose(x[0], x1),
            4: lambda x: np.isclose(x[1], y1)
        }

        self.mark_boundary(markings)

class LineMesh(Mesh):
    """
    Create a 1D line mesh between x0 and x1 with a given element size.
    Parameters:
    -------------
        x0: Start point of the mesh.
        x1: End point of the mesh.
        dx: Element size.
        comm: MPI communicator, default is MPI.COMM_WORLD.
        **kwargs: Additional keyword arguments passed to dolfinx.mesh.create_interval.
    """
    def __init__(self, x0, x1, dx, comm=MPI.COMM_WORLD, **kwargs):
        self.comm = comm

        # Create the mesh
        self.msh = _cartesian_mesh([x0], [x1], [dx], comm, **kwargs)
        markings = {
            1: lambda x: np.isclose(x[0], x0),
            2: lambda x: np.isclose(x[0], x1)
        }
        self.mark_boundary(markings)
