import numpy as np

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()

def build_field_scalar_function(domain_mesh, fictitious_mesh, inside_value, outside_value, name=None):
    """
    Create a scalar function defined on the domain mesh that is constant inside
    the fictitious mesh and has a different value outside.
    Parameters
    ----------
    domain_mesh : flatironx.mesh
        The mesh of the domain where the function is defined.
    fictitious_mesh : flatironx.mesh
        The mesh of the fictitious region where the function takes the inside value.
    inside_value : float
        The value of the function inside the fictitious region.
    outside_value : float
        The value of the function outside the fictitious region.
    name : str, optional
        The name of the function. If not provided, name will default to "field_scalar_function".

    Returns
    -------
    dolfinx.fem.Function
        A function defined on the domain mesh, where each local degree of freedom (DOF) is assigned
        the inside value if it is in the fictitious region, and the outside value otherwise.

    """
    V = dolfinx.fem.functionspace(domain_mesh.msh, ('CG', 1))
    v = dolfinx.fem.Function(V) # Initialize the function (0 by default)

    Q = dolfinx.fem.functionspace(fictitious_mesh.msh, ('CG', 1))
    q = dolfinx.fem.Function(Q)
    q.x.array[:] = inside_value

    cell_map = domain_mesh.msh.topology.index_map(domain_mesh.msh.topology.dim)
    cell_map = domain_mesh.msh.topology.index_map(domain_mesh.msh.topology.dim)
    num_cells_on_proc = cell_map.size_local + cell_map.num_ghosts
    cells = np.arange(num_cells_on_proc, dtype=np.int32)
    interpolation_data = dolfinx.fem.create_interpolation_data(V, Q, cells, padding=1e-8)

    v.interpolate_nonmatching(q, cells, interpolation_data=interpolation_data)

    v_array = v.x.array
    mask = np.abs(v_array - inside_value) > 1e-10  # tolerance to avoid fp noise
    v_array[mask] = outside_value
    v.x.scatter_forward()

    if name is not None:
        v.name = name

    else:
        v.name = 'field_scalar_function'

    return v

def build_rank_indicator_function(mesh, name=None):
    """
    Create a scalar function that indicates the rank of the process on which it is defined.
    Parameters
    ----------
    mesh : flatironx.mesh
        The mesh on which the function is defined.
    name : str, optional
        The name of the function. If not provided, name will default to "PID".
    Returns
    -------
    dolfinx.fem.Function
        A function defined on the mesh, where each local degree of freedom (DOF) is assigned the rank ID.
    """
    V = dolfinx.fem.functionspace(mesh.msh, ('CG', 1))
    f = dolfinx.fem.Function(V)

    # Assign each local DOF the current rank's ID
    rank = mesh.msh.comm.rank
    f.x.array[:] = np.full_like(f.x.array, rank, dtype=f.x.array.dtype)

    f.x.scatter_forward()

    if name is not None:
        f.name = name

    else:
        f.name = 'PID'

    return f




