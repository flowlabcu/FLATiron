import dolfinx 
import numpy as np
import ufl

def interpolate_nonmatching(u, V):
    """
    Project a function u onto a function space V using interpolation.
    Parameters
    ----------
    u : dolfinx.fem.Function
        The function to be projected.
    V : dolfinx.fem.FunctionSpace
        The function space onto which u is projected.
    
    Returns
    -------
    dolfinx.fem.Function
        The projected function in the function space V.
    """
    v = dolfinx.fem.Function(V)
    cell_map = V.mesh.topology.index_map(V.mesh.topology.dim)
    num_cells_on_proc = cell_map.size_local + cell_map.num_ghosts
    cells = np.arange(num_cells_on_proc, dtype=np.int32)
    interpolation_data = dolfinx.fem.create_interpolation_data(V, u.function_space, cells=cells, padding=1e-8)
    v.interpolate_nonmatching(u, cells=cells, interpolation_data=interpolation_data)
    v.x.scatter_forward()
    return v