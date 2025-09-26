import adios4dolfinx
import basix
import dolfinx

from mpi4py import MPI
from pathlib import Path

def _bp_get_mesh(filepath:Path):
    # Ensure filepath is a Path object (adios4dolfinx is buggy with str)
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    bp_file = filepath

    return adios4dolfinx.read_mesh(bp_file, comm=MPI.COMM_WORLD)

def _build_function_space(msh:dolfinx.mesh.Mesh, element_family:str='CG', element_degree:int=1, element_shape=None):
        if element_shape == 'scalar' or element_shape is None:
            _shape = ()
        elif element_shape == 'vector':
            _shape = (msh.topology.dim, )
        elif element_shape == 'tensor':
            _shape = (msh.topology.dim, msh.topology.dim)
        else:
            raise ValueError(f"Unknown element_shape: {element_shape}. Must be one of 'scalar', 'vector', or 'tensor'.")
    
        element = basix.ufl.element(element_family, msh.basix_cell(), element_degree, shape=_shape)
        V = dolfinx.fem.functionspace(msh, element)
        return V

def bp_get_time_steps(filepath:Path, name:str=None):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    bp_file = filepath

    return adios4dolfinx.read_timestamps(bp_file, comm=MPI.COMM_WORLD, function_name=name)

def bp_read_function(filepath:str|Path, time_id:int=0, name:str=None,
                     element_family:str='CG', element_degree:int=1, element_shape=None):
    """
    Read a function from a BP file.
    Parameters
    ----------
    filepath : str or Path
        Path to the BP file.
    time_id : int, optional
        Time step index to read. Default is 0 (first time step).
    name : str, optional
        Name of the function to read. If None, the first function in the file is read.
    element_family : str, optional
        Finite element family. Default is 'CG'.
    element_degree : int, optional
        Finite element degree. Default is 1.
    element_shape : str, optional
        Shape of the element. Can be 'scalar', 'vector', or 'tensor'. Default is None (scalar).
    Returns
    -------
    f : dolfinx.fem.Function
        The function read from the BP file.
    """
    # Ensure filepath is a Path object (adios4dolfinx is buggy with str)
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    bp_file = filepath
    
    # Read the mesh from the BP file and build the function space
    msh = _bp_get_mesh(filepath)
    V = _build_function_space(msh, element_family, element_degree, element_shape)

    # Get time step 
    time_step = bp_get_time_steps(bp_file, name=name)[time_id]

    # Read the function from the BP file
    f = dolfinx.fem.Function(V)
    f.name = name
    adios4dolfinx.read_function(bp_file, f, time=time_step, name=name)
    f.x.scatter_forward()

    return f

def bp_to_pvd(bp_filepath:str, pvd_filepath:str,
              name:str, time_id:str|int=None, 
              element_family:str='CG', element_degree:int=1, element_shape=None):
    """
    Convert a BP file to a PVD file for visualization in Paraview.
    Parameters
    ----------
    bp_filepath : str or Path
        Path to the BP file.
    pvd_filepath : str or Path
        Path to the output PVD file.
    name : str
        Name of the function to read from the BP file.
    time_id : int or 'all', optional
        Time step index to read. If 'all', all time steps are read. Default is None (first time step).
    element_family : str, optional
        Finite element family. Default is 'CG'.
    element_degree : int, optional
        Finite element degree. Default is 1.
    element_shape : str, optional
        Shape of the element. Can be 'scalar', 'vector', or 'tensor'. Default is None (scalar).
    
    Returns
    -------
    None    
    """
    # Ensure filepath is a Path object (adios4dolfinx is buggy with str)
    if not isinstance(bp_filepath, Path):
        bp_filepath = Path(bp_filepath)
    bp_file = bp_filepath

    if not isinstance(pvd_filepath, Path):
        pvd_filepath = Path(pvd_filepath)
    pvd_file = pvd_filepath.with_suffix('.pvd')

    # If time_id is an integer, read that time step. If 'all', read all time steps.
    if isinstance(time_id, int):
        f = bp_read_function(bp_filepath, time_id=time_id, name=name,
                             element_family=element_family, element_degree=element_degree, element_shape=element_shape)
        
        with dolfinx.io.VTKFile(f.function_space.mesh.comm, pvd_file, 'w') as vtk:
            vtk.write_function(f, bp_get_time_steps(bp_file, name=name)[time_id])
    
    elif time_id == 'all':
        with dolfinx.io.VTKFile(MPI.COMM_WORLD, pvd_file, 'w') as vtk:
            for t, i in enumerate(bp_get_time_steps(bp_file, name=name)):
                if MPI.COMM_WORLD.rank == 0:
                    print(f'Saving function {name} at time step {t} to file {pvd_file}')
                f = bp_read_function(bp_filepath, time_id=t, name=name,
                                     element_family=element_family, element_degree=element_degree, element_shape=element_shape)
                
                vtk.write_function(f, t)

def bp_to_xdmf(bp_filepath:str, xdmf_filepath:str,
              name:str, time_id:str|int=None, 
              element_family:str='CG', element_degree:int=1, element_shape=None):
    """
    Convert a BP file to an XDMF file for visualization in Paraview.
    Parameters
    ----------
    bp_filepath : str or Path
        Path to the BP file.
    xdmf_filepath : str or Path
        Path to the output XDMF file.
    name : str
        Name of the function to read from the BP file.
    time_id : int or 'all', optional
        Time step index to read. If 'all', all time steps are read. Default is None (first time step).
    element_family : str, optional
        Finite element family. Default is 'CG'.
    element_degree : int, optional
        Finite element degree. Default is 1.
    element_shape : str, optional
        Shape of the element. Can be 'scalar', 'vector', or 'tensor'. Default is None (scalar).     
    Returns 
    -------
    None
    """
    # Ensure filepath is a Path object (adios4dolfinx is buggy with str)
    if not isinstance(bp_filepath, Path):
        bp_filepath = Path(bp_filepath)
    bp_file = bp_filepath

    if not isinstance(xdmf_filepath, Path):
        xdmf_filepath = Path(xdmf_filepath)
    xdmf_file = xdmf_filepath.with_suffix('.xdmf')

    # If time_id is an integer, read that time step. If 'all', read all time steps.
    if isinstance(time_id, int):
        f = bp_read_function(bp_filepath, time_id=time_id, name=name,
                             element_family=element_family, element_degree=element_degree, element_shape=element_shape)
        
        with dolfinx.io.XDMFFile(f.function_space.mesh.comm, xdmf_file, 'w') as xdmf:
            xdmf.write_mesh(f.function_space.mesh)
            xdmf.write_function(f, bp_get_time_steps(bp_file, name=name)[time_id])  
    
    elif time_id == 'all':
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, xdmf_file, 'w') as xdmf:
            xdmf.write_mesh(_bp_get_mesh(bp_filepath))
            for t, i in enumerate(bp_get_time_steps(bp_file, name=name)):
                if MPI.COMM_WORLD.rank == 0:
                    print(f'Saving function {name} at time step {t} to file {xdmf_file}')
                f = bp_read_function(bp_filepath, time_id=t, name=name,
                                     element_family=element_family, element_degree=element_degree, element_shape=element_shape)
                
                xdmf.write_function(f, t)