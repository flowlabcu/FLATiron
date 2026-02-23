import dolfinx
import os
import xml.etree.ElementTree as ET

from mpi4py import MPI
class PVDWriter:
    """
    Simple .pvd (VTK Collection) writer that stores filenames (relative) per timestep.
    Parameters
    ----------
    filename : str
        Path to the output .pvd file.
    """
    def __init__(self, filename):
        self.filename = filename
        self.collection = ET.Element(
            "VTKFile",
            type="Collection",
            version="0.1",
            byte_order="LittleEndian",
        )
        self.dataset_collection = ET.SubElement(self.collection, "Collection")
        self.datasets = []

    def add_dataset(self, timestep, file):
        """
        Add a dataset to the collection.
        Parameters
        ----------
        timestep : float
            The time step associated with the dataset.
        file : str
            The path to the dataset file.
        """
        # store only the basename so the pvd can sit in the same folder as the files
        dataset = ET.SubElement(
            self.dataset_collection,
            "DataSet",
            timestep=str(timestep),
            group="",
            part="0",
            file=os.path.basename(file),
        )
        self.datasets.append(dataset)

    def write(self):
        """
        Write the .pvd file to disk.
        """
        tree = ET.ElementTree(self.collection)
        # ensure target dir exists
        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
        tree.write(self.filename, encoding="utf-8", xml_declaration=True)

def read_mesh(mesh_file, gdim=None, comm=MPI.COMM_WORLD):
    """
    Read a mesh from a file with dolfinx.

    Parameters
    -----------
        mesh_file : str, Path to the mesh file.
        gdim : int, Geometric dimension of the mesh. Default is None.
        comm : MPI communicator. Default is MPI.COMM_WORLD.

    Returns
    -----------
        mesh : dolfinx.mesh.Mesh, The mesh object.
        subdomain : dolfinx.mesh.MeshTags, The subdomain tags.
        boundary : dolfinx.mesh.MeshTags, The boundary tags.
    """

    if mesh_file.lower().endswith('.msh'):
        return _read_gmsh(mesh_file, gdim=gdim, comm=comm)
    
    elif mesh_file.lower().endswith('.xdmf'):
        return _read_xdmf(mesh_file, comm=comm)
    
    elif mesh_file.lower().endswith('.h5'):
        raise ValueError(f'.h5 no longer supported. Use .xdmf instead.')
    
    else:
        raise ValueError(f"Unsupported mesh file format: {mesh_file}. Supported formats are .msh and .xdmf.")
    
def _read_gmsh(mesh_file, gdim=None, comm=MPI.COMM_WORLD):  
    """
    Read a mesh from a GMSH file with dolfinx.

    Parameters
    -----------
        mesh_file : str, Path to the mesh file.
        gdim : int, Geometric dimension of the mesh. Default is None.
        comm : MPI communicator. Default is MPI.COMM_WORLD.

    Returns
    -----------
        mesh : dolfinx.mesh.Mesh, The mesh object.
        subdomain : dolfinx.mesh.MeshTags, The subdomain tags.
        boundary : dolfinx.mesh.MeshTags, The boundary tags.
    """

    # If the geometric dimension is given, we return exactly the read in mesh
    if gdim is not None:
        return dolfinx.io.gmshio.read_from_msh(mesh_file, comm, gdim=gdim)
    
    # If the geometric dimension is not given, we read the mesh and attempt to guess the
    # geometric dimension from the mesh file

    (_mesh, _subdomain, _boundary) = dolfinx.io.gmshio.read_from_msh(mesh_file, comm)

    # If the topological dimension is 3, the gdim is also 3
    tdim = _mesh.topology.dim
    if tdim == 3:
        return (_mesh, _subdomain, _boundary)
    
    # If the topological dimension is 2, assume the user wants gdim = tdim 
    return dolfinx.io.gmshio.read_from_msh(mesh_file, comm, gdim=tdim)

def _read_xdmf(mesh_file, comm=MPI.COMM_WORLD):
    """
    Read a mesh from a XDMF file with dolfinx.

    Parameters
    -----------
        mesh_file : str, Path to the mesh file.
        comm : MPI communicator. Default is MPI.COMM_WORLD.

    Returns
    -----------
        mesh : dolfinx.mesh.Mesh, The mesh object.
        subdomain : dolfinx.mesh.MeshTags, The subdomain tags.
        boundary : dolfinx.mesh.MeshTags, The boundary tags.
    """
    with dolfinx.io.XDMFFile(comm, mesh_file, 'r') as fid:
        mesh = fid.read_mesh()
        mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
        subdomain = fid.read_meshtags(mesh, 'cell_tags')
        boundary = fid.read_meshtags(mesh, 'facet_tags')

    return (mesh, subdomain, boundary)

