import flatiron_tk
import numpy as np
import pytest
import ufl 

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()
MPI = import_mpi4py()

from flatiron_tk.physics import *
from flatiron_tk.mesh import LineMesh
from flatiron_tk.solver import NonLinearSolver, NonLinearProblem, ConvergenceMonitor


def test_scalar_transport():
    """
    Solve 1D steady advection–diffusion–reaction with Dirichlet BCs
    and compare with the analytical solution.
    """

    # Mesh

    num_elements = 1024
    line_mesh = LineMesh(0, 1, 1 / num_elements)  # 128 elements
    st = SteadyScalarTransport(line_mesh, "a")
    st.set_element("CG", 1)
    st.build_function_space()

    # Parameters
    u = 1.0
    h = 1 / num_elements
    Pe = 5
    D = u / Pe * (h / 2)  # diffusivity
    r = 1.0

    print(Pe)

    st.set_advection_velocity(u)
    st.set_diffusivity(D)
    st.set_reaction(r)

    # Weak form + stabilization
    st.set_weak_form()
    st.add_stab("shakib")

    # Boundary conditions
    bc_dict = {
        1: {"type": "dirichlet", "value": flatiron_tk.constant(line_mesh, 0.0)},
        2: {"type": "dirichlet", "value": flatiron_tk.constant(line_mesh, 0.0)},
    }
    st.set_bcs(bc_dict)

    # Solver
    def my_custom_ksp_setup(ksp):
        ksp.setType(ksp.Type.FGMRES)
        ksp.pc.setType(ksp.pc.Type.LU)
        ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)
        ksp.setMonitor(ConvergenceMonitor("ksp"))

    problem = NonLinearProblem(st)
    solver = NonLinearSolver(
        line_mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup
    )
    solver.solve()

    # ----------------------------------------------------------------------
    # Compare numerical vs analytical solution at DOF coordinates
    # ----------------------------------------------------------------------
    uh = st.get_solution_function()

    g = u / D
    def exact_fun(x):
        # x is a (3, N) array
        xvals = x[0]
        exp_neg_g = np.exp(-g)
        return (1/u) * (
            xvals - (exp_neg_g - np.exp((xvals - 1)*g)) / (exp_neg_g - 1)
        )
        

    u_exact = dolfinx.fem.Function(st.get_function_space())
    u_exact.interpolate(exact_fun)

    error_form = dolfinx.fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx)
    error_L2 = np.sqrt(dolfinx.fem.assemble_scalar(error_form))

    assert error_L2 < 1e-3

