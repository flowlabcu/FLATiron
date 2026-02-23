# FLATironX-Dev Meeting Notes

## 14-04-2025
- "I don't know what's going on in the Github"
- Going over branches 

Branch strategy: 
    - Feature branches only! No "FlATironX-Name"
    - Example: "new_X-mesh_class" 

## 12-05-2025
- Pull request strat: Do pull request for features, merge changes at FLATironX-Dev meeting!
- Discussed io, mesh, basic_mesh, problem, solver, physics_problem classes
- Goals: 
    - Transient scalar transport. Update all setter/getter defs to set_ or get_ for consistency 
    - Classes to be names Steady or Transient for consistency 

- Begin weekly meetings for summer development 

## 19-05-2025
- Covered physics problem to scalar transport 
- Physics Class Order of Attack:

    Immediate Deliverables:
    1. Steady Scalar Transport (Done)
    2. Multiphysics Problem --- Transient Scalar 
    3. Steady Stokes --- Steady Navier-Stokes
    4. Unsteady Stokes --- Unsteady Navier-Stokes  

    Future Projects:

    5. Non-Newtonian Fluids (power law + )
    6. Linear Elastodynamics 
    7. ALE Support

- Jess: Multiphysics Problem 
- Nick: Transient STP 

# 30-05-2025
- Transient ADR done 
- Jess starts multiphysics problem && check in on Wednesday 
- Fictitious domain / field scalar -- Nick 
- Use dolfin 0.9.0 until FLATironX first release, then update to dolfin 0.10.0 (or latest version)

# 23-06-2025
- Steady multiphysics finished and tested with 2d ADR canonical problem 
- Jess does Transient Multiphysics (check in on July 1) 
- Nick does Steady Stokes (maybe Steady Navier-Stokes)

# 14-07-2025
- Transient multiphysics complete - need to test with true monolithic solve (i.e. Bousinesq Approx)

## Several dolfinx quirks to note:
### Subspaces and boundary conditions: 
Two dolfinx types can be passed into `dolfinx.fem.dirichletbc()`: `dolfinx.fem.Function` and `dolfinx.fem.Constant`. If we are solving a `mixed-element` problem, we need to decide if we are to collapse the subspace and provide a map back to the original mixed function space.

**If we are passing in the `Constant` type:**
- We **do not** collapse the function space and pass a map back to the larger function space when defining DOFs. (Note that we may sometimes have to concatinate the dofs depending on how they were built from the subspace)

    ``` 
    if isinstance(bc_val, dolfinx.fem.Constant):
        dofs = dolfinx.fem.locate_dofs_topological(function_space, mesh.msh.topology.dim - 1, mesh.boundary.find(bnd_id))

        if isinstance(dofs, list):
            # For constants/ndarrays, flatten dofs
            dofs = np.concatenate(dofs)

        bc = dolfinx.fem.dirichletbc(bc_val, dofs, function_space)
    ```

- Here the function space can be `dolfinx.fem.functionspace` or `dolfinx.fem.functionspace.sub(i)`
- _Additionally note that vector valued boundary conditions **cannot** be defined as a `Constant` type... for some reason_

**If we are passing a `Function` type:**
- We must decide if we can find the DOFs directly from the function space (as in the case of a non-mixed function space), or if we need to provide the subspace and a map to the original space. We have found the best way of generalizing this is by first detecting if the function space we pass to `dolfinx.fem.locate_dofs_topological` is a subspace. This is done by evaulating the length of the `component` attribute. 

    ```
    def is_subspace(function_space):
        if len(function_space.component()) == 0: # 
            return False 
        else:
            return True
    ```
    - If the function space is a subspace, we then create a tuple that passes the mapping back to the original space. 
    ```
    def get_function_space_search(function_space)
        if is_subspace(function_space):
        V_sub, V_submap = function_space.collapse()
        return (V, V_sub)
        else:
            return function_space
    ```

- We then find the dofs as normal:

    ```
    _function_space_search = get_function_space_search(function_space)
        dofs = dolfinx.fem.locate_dofs_topological(_function_space_search, mesh.msh.topology.dim - 1, mesh.boundary.find(bnd_id))
    ```

- Finally, if the function space is a vector (or tensor) function space, the dofs will return as a list. In this instance, we provide the diricletbc builder with the function space:

    ```
    if isinstance(dofs, list):
        bc = dolfinx.fem.dirichletbc(bc_val, dofs, function_space)
    else:
        bc = dolfinx.fem.dirichletbc(bc_val, dofs)
    ```

### Divergence term in the Stokes/Navier-Stokes Equations
The divergence term 
    
$
T3 = q * \nabla \cdot \underline{\textbf{u}} 
$

must have the same sign as the PSPG residual 

$
PSPG = \frac{1}{\rho} \tau \nabla q \cdot R 
$

where $R$ is the residual. 

### Next steps
**Ideally by August 1**
1. Add free convection demo 
2. Add block solver object 
3. Add Indicator Fields / Field Scalar Objects 
4. Copy documentation over from LEGACY and update/make changes 
5. Clean up the demos and add comments 
6. Add set/get for residuals 

**By September 1**
1. Unit tests

**Cont Development**
1. Quality of Life (Ex)
    - flatironx.Constant(val)
    - Inlet profiles (parabola, oid)
    - RCR boundary conditions
    - QR boundary conditions 

# 04/08/2025
## DOLFINx Notes
1. Scalar (probably vector and tensor) assemebly does not automatically reduce across all processes.

In **LEGACY** dolfin: 
    
    a = fe.asseble(form)
    
    # RETURNS:
    a = MPI.COMM_WORLD.allreduce(a, op=MPI.SUM) 
    

In **DOLFINx**, this operation needs to be done explicitly by the user:

    a = dolfinx.fem.assemble_scalar(form)
    
    # RETURNS 
    a = a_val_0  # On PID 0 
    # and 
    a = a_val_1  # On PID 1 

    # To maintain expected output 
    a = dolfinx.fem.assemble_scalar(form)
    a = MPI.COMM_WORLD.allreduce(a, op=MPI.SUM)

# 02/09/2025
- Update Jess on 
    1. Block solver
    2. adios4dolfinx
    3. brinkman formulation

- Next Steps:
    1. Update Mesh module setters/getters (Jess)
    2. Update testing suite (Jess)
    3. Update documentation (Nick and Jess)

- Possible steps before release
    1. foam2dolfin
    2. Linear/hyperelastic modules (Michael)
    3. Parabaloid inlet condition 

# 02/10/2025 - Notes from FLATironX release 
- For building the ReadTheDocs: 
    - RTD builds a Docker container to install the library and build the docs website. This means that all the relevant 
    dependencies need to be "installed" into the Docker container. Since `adios4dolfinx`, `dolfinx`, `MPI`, `PETSc`, `basix`, etc. are
    all huge or conda installs only, we have to fake the installation for the docs page to build. We can use mock imports in 
    `docs/source/conf.py` to do this: 

    ```
    autodoc_mock_imports = [
    "dolfinx",
    "basix",
    "adios4dolfinx",
    "ufl",
    "petsc4py",
    "mpi4py",
    ]
    ```

    - For the release, we thought we had to do the import using a `info_messages` module that mocked the imports. This is how it was 
    done for the FLATiron legacy / feFlow release. **This does not seem to be the case** the `info_messages` is not needed so long the imports are mocked correctly. For that reason, I am not updating the dev branches with info_messages.    

    - I have updated the `main` branch with `docs`, `tests`, and docstrings to reflect the changes we made in the `release_branch` on Github. 

    - **FOR THE NEXT RELEASE** we will actually include a version number. 





