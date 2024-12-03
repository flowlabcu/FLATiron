'''
Wrapper for FEniCS mesh containing mesh and boundary information
'''

import os

# ------------------------------------------------------- #

from ..info.messages import import_fenics
fe = import_fenics()

from ..io import *

def _load_mesh(comm, mesh_file):
    hdf = fe.HDF5File(comm, mesh_file, 'r')
    mesh = fe.Mesh(comm)
    hdf.read(mesh, 'mesh', False)
    boundary = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    hdf.read(boundary, 'boundaries')
    hdf.close()
    return mesh, boundary

class Mesh():

    """This class is a wapper object for dolfin.cpp.Mesh.

    :param mesh_file: Path to the .h5 file containing the mesh
    :type mesh: str

    :param mesh: The FEniCS mesh object.
    :type mesh: fe.Mesh

    :param boundary: The fenics MeshFunction object with marked boundary values (dim = mesh.dim - 1)
    :type boundary: fe.MeshFunction

    This class can be instantiate in 2 ways. Either by supplying *only* the ``mesh_file`` or *both* the FEniCS ``mesh`` and ``boundary`` objects

    :Example:


        .. code-block:: python

            # Instantiate with an existing *.h5 file format
            # This Mesh class will automatically look for the `boundary` group in the hdf5 file
            Mesh(mesh_file=\"mesh_file.h5\")


        .. code-block:: python

            # Instantiate with the fenics mesh and boundary objects
            import fenics as fe
            fenics_mesh = fe.UnitSquareMesh(4, 4)
            fenics_boundary = fe.MeshFunction(\"size_t\", fenics_mesh, fenics_mesh.topology().dim()-1)
            Mesh(mesh=fenics_mesh, boundary=fenics_boundary)

    """

    def __init__(self, **kwargs):

        # Get comm
        self.comm = kwargs.get('comm', fe.MPI.comm_world)

        # Get mesh
        _mesh = kwargs.get('mesh', None)
        _mesh_file = kwargs.get('mesh_file', None)
        if _mesh is not None:
            _boundary = kwargs.get('boundary', None)
            self.mesh = _mesh
            if _boundary is None:
                self.boundary = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
            else:
                self.boundary = _boundary
        elif _mesh_file is not None:
            self.mesh, self.boundary = _load_mesh(self.comm, _mesh_file)
        else:
            raise ValueError('All mesh inputs are None. Please provide either the mesh or mesh file')

        # Get mpi variables
        self.mpi_rank = self.comm.rank
        self.mpi_size = self.comm.size

        # Get mesh dimension
        self.dim = self.mesh.topology().dim()

        # boundary normals
        self._flat_boundary_normals = {}

    def fenics_mesh(self):
        """
        Return the original fenics mesh
        """
        return self.mesh

    def mark_boundary(self, boundary_id, eval_func, *args):
        """
        Mark a boundary on a mesh with boundary_id based on eval_func
        eval_func(x, *args) should return a boolean
        """
        class BD(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and eval_func(x, *args)
        BD().mark(self.boundary, boundary_id)

    def cell_diameter(self):
        """
        Return the ufl object for mesh cell diameter
        """
        return fe.CellDiameter(self.mesh)

    def facet_normal(self):
        """
        Return the ufl object for mesh facet norma
        """
        return fe.FacetNormal(self.mesh)

    def mean_boundary_normal(self, boundary_id):
        """
        \\overline{\\hat{n}} = \\frac{\\int_{\\Gamma} \\hat{n} d\\Gamma}{\\int_{\\Gamma} d\\Gamma}
        """
        # mesh = mesh.mesh; dim = mesh.dim; boundary = physics.ds(boundary_id)
        
        # Calls for facet normal of one side?
        n = self.facet_normal() 
        ds = fe.ds(subdomain_data=self.boundary) # what dis
        normal = np.array([fe.assemble(n[i] *  ds(boundary_id)) for i in range(self.dim)])
        normal_mag = np.linalg.norm(normal, 2)
        normal_vector = (1/normal_mag) * normal
        return normal_vector 

    def write(self, mesh_file):
        """
        Writes the mesh into either pvd or h5 format.

        If `mesh_file` ends with "pvd", writes to a ParaView VTU file.
        If `mesh_file` ends with "h5", writes to an HDF5 file.

        :param mesh_file: The path to the output file.
        :type mesh_file: str

        :raises ValueError: If `mesh_file` has an unsupported extension.

        :Example:

        Writing mesh to ParaView VTU format::

            mesh.write("output.pvd")

        Writing mesh to HDF5 format::

            mesh.write("output.h5")
        """
        if mesh_file.endswith('pvd'):
            fe.File(mesh_file) << self.mesh
            fe.File(mesh_file[:-4]+'_boundary.pvd') << self.boundary
        elif mesh_file.endswith('h5'):
            hdf = h5_init_output_file(mesh_file, mesh=self.mesh, boundaries=self.boundary)
            hdf.close()

    def flat_boundary_normal(self, boundary_id):

        """
        Return inwards boundary normal associated with the boundary_id
        NOTE that this function assumes that your face boundary is flat, i.e.,
        all facets on the boundary has the same normal direction
        """

        if len(self._flat_boundary_normals) == 0:
            self._build_flat_boundary_normals()
        return self._flat_boundary_normals[boundary_id]

    def _build_flat_boundary_normals(self):

        # Init connectivity data for the boundary
        fe.FunctionSpace(self.mesh, 'CG', 1)

        # Build search ids
        unique_ids = list(np.unique(self.boundary.array()))

        # Gather all search ids
        all_unique_ids = self.comm.allgather(unique_ids)
        exclude_ids = [0] # 0 is usually the interior facets, so we exclude it
        search_ids_set = set()
        for r in range(self.comm.size):
            for i in all_unique_ids[r]:
                if i not in exclude_ids:
                    search_ids_set.add(i)

        # Init normal boundary dicts
        bnd_normal_dict = {}
        for bnd_id in search_ids_set:
            bnd_normal_dict[bnd_id] = None

        # Search for normals on each boundary
        while unique_ids:

            # Boundary id to search
            bnd_id = unique_ids.pop()

            # If this unique id is not in the search ids, continue
            if bnd_id not in search_ids_set:
                continue

            # Build boundary id dictionary
            for facet in fe.facets(self.fenics_mesh()):

                # Early exit for non exterior facets
                if not facet.exterior():
                    continue

                # Assign normal
                if self.boundary.array()[facet.index()] == bnd_id:
                    bnd_normal_dict[bnd_id] = -1*facet.normal()[:]
                    break

        # gather boundaries
        all_bnd_dict = self.comm.allgather(bnd_normal_dict)
        global_bnd_normal_dict = {}
        for bnd_dict in all_bnd_dict:
            for bnd_id in bnd_dict:
                if bnd_dict[bnd_id] is not None:
                    global_bnd_normal_dict[bnd_id] = bnd_dict[bnd_id]
        self._flat_boundary_normals = global_bnd_normal_dict

    def facet(self, i):
        return fe.Facet(self.fenics_mesh(), i)

    def cell(self, i):
        return fe.Cell(self.fenics_mesh(), i)

    def avg_cell_diameter(self):
        hs = [c.h() for c in fe.cells(self.fenics_mesh())] 
        h_sum = np.sum(hs)
        num_cell = len(hs)
        h_sum = self.comm.allgather(h_sum)
        num_cell = self.comm.allgather(num_cell)
        h_avg = np.sum(h_sum)/np.sum(num_cell)
        return h_avg



