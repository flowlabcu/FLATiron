import numpy as np

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()
MPI = import_mpi4py()

from ..io import *
class Mesh():
    """
    A base class for creating and managing computational meshes using Dolfinx.  
    
    Parameters
    -----------
        comm: MPI communicator, default is MPI.COMM_WORLD
        mesh: dolfinx mesh object, default is None
        mesh_file: mesh file name, default is None
        boundary: dolfinx mesh tags for boundary, default is None
        subdomain: dolfinx mesh tags for subdomain, default is None
        gdim: geometric dimension of the mesh, optional, used when loading from file
    """
    def __init__(self, **kwargs):
        self.comm = kwargs.get('comm', MPI.COMM_WORLD)

        # --- Get the mesh -- - #
        _msh = kwargs.get('mesh', None)  
        _msh_file = kwargs.get('mesh_file', None) 

        # --- Load fenics mesh objects --- #
        if _msh is not None:
            # Assigm dolfinx mesh object to flationx mesh object
            self.msh = _msh 

            # Set empty boundary and subdomain if not provided, otherwise use the provided ones
            _boundary = kwargs.get('boundary', None)
            _subdomain = kwargs.get('subdomain', None)

            if _boundary is None:
                self.boundary = dolfinx.mesh.meshtags(self.msh, self.get_fdim(), np.array([]), np.array([]))
            else:
                self.boundary = _boundary

            if _subdomain is None: 
                self.subdomain = dolfinx.mesh.meshtags(self.msh, self.get_tdim(), np.array([]), np.array([]))
            else: 
                self.subdomain = _subdomain

        elif _msh_file is not None:
            # Option to provide gdim instead of having the mesh reader guess it
            _gdim = kwargs.get('gdim', None)
            self.msh, self.subdomain, self.boundary = io.read_mesh(_msh_file, gdim=_gdim, comm=self.comm)

    def get_tdim(self):
        """
        Returns the topological dimension of the mesh.
        """
        return self.msh.topology.dim

    def get_fdim(self):
        """
        Returns the facet dimension of the mesh; the dimension of the mesh boundary.
        """
        return self.msh.topology.dim - 1
    
    def get_gdim(self):
        """
        Returns the geometric dimension of the mesh.
        """
        return self.msh.geometry.dim
    
    def get_cell_diameter(self):
        """
        Returns the ufl cell diameter of the mesh.
        """
        return ufl.CellDiameter(self.msh)
    
    def get_facet_normal(self):
        """
        Returns the ufl facet normal of the mesh.
        """
        return ufl.FacetNormal(self.msh)

    def _mark_entities(self, marking_dict, entity_dim):
        """
        Mark entities (facets or cells) of the mesh based on user-defined markers. Method called by mark_boundary and mark_subdomain.
        Parameters
        -----------
            marking_dict : dict, A dictionary where keys are marker ids and values are functions that return boolean arrays
                        indicating which entities to mark.
            entity_dim : int, The dimension of the entities to mark (e.g., facets or cells).
        Returns
        -------
            dolfinx.mesh.MeshTags: A MeshTags object containing the marked entities.
        """

        # Build connectivity
        if entity_dim != self.get_tdim():
            self.msh.topology.create_connectivity(entity_dim, self.get_tdim())

        # Create a list of entities and marking ids
        entity_ids = []
        all_marking_ids = []
        
        # for marker, idx in zip(marking_function_lst, marking_ids):
        for idx, marker in marking_dict.items():
            this_bnd = dolfinx.mesh.locate_entities(self.msh, entity_dim, marker)
            this_ids = [idx for i in range(len(this_bnd))]
            entity_ids.extend(this_bnd)
            all_marking_ids.extend(this_ids)

        # Sort marking and entities by entity ids
        entity_ids = np.array(entity_ids)
        all_marking_ids = np.array(all_marking_ids)
        sorted_ids = np.argsort(entity_ids)
        entity_ids = entity_ids[sorted_ids]
        all_marking_ids = all_marking_ids[sorted_ids]

        return dolfinx.mesh.meshtags(self.msh, entity_dim, entity_ids, all_marking_ids)

    def mark_boundary(self, marking_dict):
        """
        Mark the boundary of the mesh with user-defined markers.
        Parameters
        -----------
            marking_dict : dict, A dictionary where keys are marker ids and values are functions that return boolean arrays
                        indicating which facets to mark.
        """
        self.boundary = self._mark_entities(marking_dict, self.get_fdim())

    def mark_subdomain(self, marking_dict):
        """
        Mark the subdomains of the mesh with user-defined markers.
        Parameters
        -----------
            marking_dict : dict, A dictionary where keys are marker ids and values are functions that return boolean arrays
                        indicating which cells to mark.
        """
        self.subdomain = self._mark_entities(marking_dict, self.get_tdim())

    def get_num_facets_local(self):
        """
        Returns the number of facets in the local mesh partition.
        """
        return self.msh.topology.index_map(self.get_tdim()-1).size_local
    
    def get_mean_boundary_normal(self, boundary_id):
        """
        Calculate the mean normal vector of a given boundary.
        Parameters
        ------------
            boundary_id (int): The id of the boundary for which to compute the mean normal.

        Returns
        ------------
            np.ndarray: A unit vector representing the mean outward normal of the boundary.
        """

        n = self.get_facet_normal()
        ds = ufl.Measure('ds', domain=self.msh, subdomain_data=self.boundary)
        

        local_normal_array = []
        for i in range(self.get_gdim()):
            form_i = dolfinx.fem.form(n[i] * ds(boundary_id))
            assemble_i = dolfinx.fem.assemble_scalar(form_i)
            local_normal_array.append(assemble_i)

        local_normal_array = np.array(local_normal_array)

    
        glabal_normal_array = self.msh.comm.allreduce(local_normal_array, op=MPI.SUM)


        norm = np.linalg.norm(glabal_normal_array, 2)
        
        normal_array = glabal_normal_array / norm

        return normal_array
    
    def get_boundary_centroid(self, boundary_id):
        """
        Calculate the centroid of a given boundary.
        Parameters
        ------------
            boundary_id (int): The id of the boundary for which to compute the centroid.

        Returns
        ------------
            np.ndarray: A point representing the centroid of the boundary.
        """
        
        # Get the facets of the boundary with the given id
        boundary_facets = self.boundary.find(boundary_id)
        facet_midpoints = []
        x = self.msh.geometry.x

        # Loop through each facet and compute the midpoint
        for facet in boundary_facets:
            facet_vertices = self.msh.topology.connectivity(self.get_fdim(), 0).links(facet)
            facet_coords = x[facet_vertices]
            facet_midpoints.append(np.mean(facet_coords, axis=0))

        # Gather midpoints from all processes
        if len(facet_midpoints) > 0:
            midpoints_local = np.array(facet_midpoints)
        else:
            midpoints_local = np.empty((0, 3))

        all_midpoints = MPI.COMM_WORLD.allgather(midpoints_local)
        centroid = None

        # If root process, compute the centroid
        if  MPI.COMM_WORLD.rank == 0:
            # Concatenate all midpoints from all processes
            all_midpoints_flat = np.concatenate(all_midpoints)
            
            # Compute the mean to get the centroid
            if len(all_midpoints_flat) > 0:
                centroid = np.mean(all_midpoints_flat, axis=0)

        # Broadcast the result to all processes
        centroid =  np.array(MPI.COMM_WORLD.bcast(centroid, root=0))
        
        return centroid
    
    def get_boundary_area(self, boundary_id):
        """
        Calculate the area of a given boundary.
        Parameters
        ------------
            boundary_id (int): The id of the boundary for which to compute the area.

        Returns
        ------------
            float: The area of the boundary.
        """
        one = dolfinx.fem.Constant(self.msh, PETSc.ScalarType(1.0))
        ds = ufl.Measure('ds', domain=self.msh, subdomain_data=self.boundary)
        form = dolfinx.fem.form(one * ds(boundary_id))
        area = dolfinx.fem.assemble_scalar(form)

        # Sum the area across all processes
        area = self.msh.comm.allreduce(area, op=MPI.SUM)

        return area

    def get_mean_cell_diameter(self):
        """
        Get mean cell diameter.
        Returns
        ---------
            float: the average length of the cell diameters within the mesh.
        """
        
        cell_diameter = self.get_cell_diameter()
        integrated_cell_diameter = dolfinx.fem.assemble_scalar(dolfinx.fem.form(cell_diameter * ufl.dx))
        volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(self.msh, dolfinx.default_scalar_type(1.0)) * ufl.dx))
        mean_cell_diameter = integrated_cell_diameter / volume
        return mean_cell_diameter

    
    

