'''
This module contains functions that load/save FEniCS mesh and functions into the hdf5 format
'''

from ..info.messages import import_fenics
fe = import_fenics()

import os
import numpy as np
import sys

def h5_init_output_file(output_file, mesh=None, boundaries=None, subdomain=None):

    """
    Initializes an HDF5 output file with the mesh, boundaries, and subdomain data. Use this function to initialize the output file
    for the simulation.

    :param output_file: Path to the .h5 file to initialize
    :type output_file: str

    :param mesh: FEniCS mesh object to write into the initial output file (Optional)
    :type mesh: fe.Mesh

    :param boundaries: The fenics MeshFunction object with marked boundary (Optional)
    :type boundaries: fe.MeshFunction

    :param subdomain: The fenics MeshFunction object with marked subdomain (Optional)
    :type subdomain: fe.MeshFunction

    :return: The initialized FEniCS hdf5 file object
    :rtype: fe.HDF5File

    """


    parent_dir = "/".join(output_file.split("/")[:-1])
    if parent_dir != '': os.makedirs(parent_dir, exist_ok=True)
    hdf = fe.HDF5File(mesh.mpi_comm(), output_file, "w")
    if mesh is not None:
        h5_write(mesh, '/mesh', h5_object=hdf)
    if boundaries is not None:
        h5_write(boundaries, '/boundaries', h5_object=hdf)
    if subdomain is not None:
        h5_write(subdomain, '/subdomains', h5_object=hdf)
    return hdf

def h5_get_group(h5_file):
    import h5py
    f = h5py.File(h5_file,'r')
    print(f.keys())

def h5_write(instance, h5_group, h5_file=None, h5_object=None, timestamp=None, mode="a"):

    """Writes FEniCS Mesh, MeshFunction, or Function object into a hdf5 file format

    :param instance: The FEniCS data you want to save.
    :type instance: fe.Mesh or fe.MeshFunction or fe.Function

    :param h5_group: HDF5 group within the HDF5 file structure to write the instance to.
    :type h5_group: str

    :param h5_file: HDF5 file name to write to. If ``h5_file`` is specified, ``h5_object`` cannot be specified
    :type h5_file: str, optional

    :param h5_object: FEniCS HDF5File object one wants to save the data to. If ``h5_object`` is specified, ``h5_file`` cannot be specified
    :type h5_object: fe.HDF5File, optional

    :param timestamp: Timestamp for the function. Used for writing time series.
    :type timestamp: float, optional

    :param mode: "a" for append, or "w" for overwrite
    :type mode: str

    """
    mesh = fe.Mesh()

    # Handle errors
    if h5_file is None and h5_object is None:
        raise KeyError('Must specify either h5_file or h5_object')
    if h5_file is not None and h5_object is not None:
        raise KeyError('Must specify either h5_file or h5_object')

    # Write
    if h5_file is not None: # Input is an h5_file

        # Check if the file already exist.
        # If it already exist, use append mode and
        # check if the dataset already exist, otherwise change to mode to write
        if os.path.isfile(h5_file) and not mode=="w":
            hdf = fe.HDF5File(mesh.mpi_comm(), h5_file, "a")
            if hdf.has_dataset(h5_group) and timestamp is None:
                raise KeyError(h5_file+" already has the dataset: "+h5_group)
        else:
            parent_dir = "/".join(h5_file.split("/")[:-1])
            if parent_dir != '': os.makedirs(parent_dir, exist_ok=True)
            hdf = fe.HDF5File(mesh.mpi_comm(), h5_file, "w")

        # Check timestmps
        if timestamp is not None:
            hdf.write(instance, h5_group, timestamp)
        else:
            hdf.write(instance, h5_group)
        hdf.close()
    elif h5_object is not None:
        if timestamp is not None:
            h5_object.write(instance, h5_group, timestamp)
        else:
            h5_object.write(instance, h5_group)

def h5_read_mesh(h5_file, mesh_name="mesh", boundaries_name="boundaries", subdomain_name="subdomains", comm=None):
    '''
    Read mesh, boundaries, and subdomains in accordance with h5init_output_file method
    Input: h5_read_mesh(h5_file, mesh_name="mesh", boundaries_name="boundaries", subdomain_name="subdomains")
    Return (mesh, boundaries, subdomains), if boundaries or subdomains do not exist, they are returned as None
    mesh must exist
    '''
    if comm is None:
        comm = fe.MPI.comm_world
    tmp = fe.Mesh()
    hdf = fe.HDF5File(comm, h5_file, "r")
    hasBoundaries = hdf.has_dataset(boundaries_name)
    hasSubdomains = hdf.has_dataset(subdomain_name)
    hdf.close()
    mesh = h5_read( h5_file, "mesh", "mesh"  )
    dim = mesh.topology().dim()

    # Check if has boundaries
    if hasBoundaries:
        boundaries = h5_read(h5_file,'boundaries','meshfunction', mesh=mesh, meshFunctionDim=dim-1)
    else:
        boundaries = None
    # Check if has subdomain
    if hasSubdomains:
        subdomains = h5_read(h5_file,'subdomains','meshfunction', mesh=mesh, meshFunctionDim=dim)
    else:
        subdomains = None
    return (mesh, boundaries, subdomains)

def h5_read(h5_file, h5_group, fe_type, mesh=None, meshFunctionDim=None, function_space=None, time_id=None):
    """Reads FEniCS Mesh, MeshFunction, or Function object from a HDF5 file

    :param h5_file: Path to the HDF5 file.
    :type h5_file: str

    :param h5_group: Name of the hdf5 file group
    :type h5_group: str

    :param fe_type: Type of data you want to save - must be 'mesh', 'meshfunction', or 'function'.
    :type fe_type: str

    :param mesh: FEniCS mesh the function or function space are defined on (required for fe_type='meshfunction').
    :type mesh: fe.Mesh

    :param meshFunctionDim: The dimension of the function_space (required for fe_type='meshfunction').
    :type meshFunctionDim: int, optional

    :param function_space: Function space the Function type is defined on (required for fe_type='function').
    :type function_space: fe.FunctionSpace optional

    :param time_id: Specify the index of the function in the time series you want to read.
    :type time_id: int, optional

    :return: The data stored in the file `h5_file` with HDF5 group=`h5_group` of type `fe_type`.

    """
    if fe_type.lower() == 'mesh':
        instance = fe.Mesh()
        hdf = fe.HDF5File(instance.mpi_comm(), h5_file, "r")
        hdf.read(instance, h5_group, False)
        hdf.close()
    elif fe_type.lower() == 'meshfunction':
        if meshFunctionDim is None:
            raise KeyError('Must specify meshFunctionDim')
        instance = fe.MeshFunction("size_t", mesh, meshFunctionDim)
        hdf = fe.HDF5File(mesh.mpi_comm(), h5_file, "r")
        hdf.read(instance, h5_group)
        hdf.close()
    elif fe_type.lower() == 'function':
        instance = fe.Function(function_space)
        hdf = fe.HDF5File(mesh.mpi_comm(), h5_file, "r")
        if time_id is None:
            hdf.read(instance, h5_group)
        else:
            hdf.read(instance, h5_group+"/vector_"+str(time_id))
        hdf.close()
    else:
        raise KeyError('Invalid h5_read fe_type input. fe_type must be mesh, function or meshfunction')

    return instance

def h5_get_time_stamp(h5_file, h5_group):
    """
    Get a list of time stamps from a specified HDF5 file and group.

    Parameters:
        h5_file (str): Path to the HDF5 file containing the data.
        h5_group (str): Name of the HDF5 group within the file where the timestamps are stored.

    Returns:
        numpy.ndarray: An array containing the time stamps extracted from the HDF5 dataset.
    """
    mesh = fe.Mesh()
    hdf = fe.HDF5File(mesh.mpi_comm(), h5_file, "r")
    nsteps = hdf.attributes(h5_group).to_dict()['count']
    tsteps = np.zeros(nsteps)
    for i in range(0,nsteps):
        tsteps[i] = hdf.attributes( h5_group+"/vector_"+str(i) ).to_dict()['timestamp']
    return tsteps

def h5_to_pvd(h5_file, h5_group, pvd_file, func_type, elem_type, elem_order, save_interval=None, mesh_h5_group='/mesh', varname="default"):

    """ Convert function data from a HDF5 file into a paraview readable file format. The HDF5 file must have the `mesh` group saved on it. This is usually a a HDF5 file that's created by the h5_init_output_file function

    :param h5_file: Path to the .h5 file.
    :type h5_file: str

    :param h5_group: Name of the h5 variable you want to convert.
    :type h5_group: str

    :param pvd_file: Path to the ParaView pvd file.
    :type pvd_file: str

    :param func_type: Either 'scalar' or 'vector'. This determines the type of function space used.
    :type func_type: str

    :param elem_order: Order of the element used to save the initial dataset. This must match the order used in the simulation!!!
    :type elem_order: int

    :param save_interval: (Optional) A Python list of 2 entries [i1, i2] where the data will be converted from time index i1 to time index i2.
    :type save_interval: list[int, int]
    """

    mesh = h5_read(h5_file, mesh_h5_group, fe_type='mesh')
    if func_type.lower() == 'vector':
        V = fe.VectorFunctionSpace(mesh, elem_type, elem_order)
    elif func_type.lower() == 'scalar':
        V = fe.FunctionSpace(mesh, elem_type, elem_order)
    elif func_type.lower() == 'tensor':
        V = fe.TensorFunctionSpace(mesh, elem_type, elem_order)
    fid = fe.File(pvd_file, 'base64')
    if varname == "default":
        varname = h5_group

    if save_interval is not None:
        timestamps = h5_get_time_stamp(h5_file, h5_group)
        if save_interval == 'all':
            iStart = 0
            iEnd = len(timestamps)
        else:
            iStart = save_interval[0]
            iEnd = save_interval[1]
        for i in range(iStart,iEnd):
            print('Converting ' + h5_file + '/' + h5_group + ' at time: ', timestamps[i])
            func = h5_read(h5_file, h5_group, fe_type='function', mesh=mesh, function_space=V, time_id=i)
            func.rename(varname, varname)
            fid.write(func, timestamps[i])
    else:
        print('Converting ' + h5_file + '/' + h5_group)
        func = h5_read(h5_file, h5_group, fe_type='function', mesh=mesh, function_space=V)
        func.rename(varname, varname)
        fid << func










