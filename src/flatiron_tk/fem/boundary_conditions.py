import numpy as np

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()

def is_subspace(V):
    '''
    Returns True if V is a subspace of a function space, False otherwise.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        The function space to check.
    Returns
    -------
    bool
        True if V is a subspace, False otherwise.
    '''
    if len(V.component()) == 0:
        return False
    else:
        return True
        
def get_function_space_search(V):
    '''
    Returns the function space to search for dofs in case V is a subspace.
    If V is not a subspace, returns V itself.
    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        The function space to check.
    Returns
    -------
    if V is a subspace
        tuple(dolfinx.fem.FunctionSpace, list mapping of the subspace to original space)
    else
        dolfinx.fem.FunctionSpace
        The function space to search for dofs.
    
    '''
    if is_subspace(V):
        V_sub, V_submap = V.collapse()
        return (V, V_sub)
    else:
        return V
    
def build_dirichlet_bc(mesh, bnd_id, bc_val, function_space):
    '''
    Builds a Dirichlet boundary condition for a given boundary and value.
    Parameters
    ----------
    mesh : flatironx.mesh 
        The mesh object containing the boundary information.
    bnd_id : int
        The boundary ID to apply the Dirichlet condition on.
    bc_val : dolfinx.fem.Function or dolfinx.fem.Constant
        The value of the Dirichlet condition, which can be a function or a constant.
    function_space : dolfinx.fem.FunctionSpace
        The function space to which the Dirichlet condition applies.
    
    Returns
    -------
    dolfinx.fem.dirichletbc : The Dirichlet boundary condition object.
    
    Raises
    ------
    TypeError : If `bc_val` is not a `dolfinx.fem.Function` or `dolfinx.fem.Constant`.
    '''

    if isinstance(bc_val, dolfinx.fem.Function):
        _function_space_search = get_function_space_search(function_space)
        dofs = dolfinx.fem.locate_dofs_topological(_function_space_search, mesh.msh.topology.dim - 1, mesh.boundary.find(bnd_id))

        # If dofs is a list, it means we have a vector function space, and we need to pass the function space
        if isinstance(dofs, list):
            bc = dolfinx.fem.dirichletbc(bc_val, dofs, function_space)
        else:
            bc = dolfinx.fem.dirichletbc(bc_val, dofs) 


    elif isinstance(bc_val, dolfinx.fem.Constant):
        dofs = dolfinx.fem.locate_dofs_topological(function_space, mesh.msh.topology.dim - 1, mesh.boundary.find(bnd_id))

        if isinstance(dofs, list):
            # For constants/ndarrays, flatten dofs
            dofs = np.concatenate(dofs)

        bc = dolfinx.fem.dirichletbc(bc_val, dofs, function_space)

    else:
        raise TypeError("bc_val must be a dolfinx.fem.Function or dolfinx.fem.Constant, got {}".format(type(bc_val)))

    return bc