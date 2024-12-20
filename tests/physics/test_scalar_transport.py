import pytest
from flatiron_tk.physics import *
from flatiron_tk.mesh import BoxMesh, RectMesh, LineMesh
import fenics as fe
from flatiron_tk.solver import PhysicsSolver
import numpy as np


def test_scalar_transport(mesh_1d):

    '''
    This test solves the problem found in the demo
    u*dc/dx - D*d^2c/dx^2 - 1 = 0
    c[0] = 0
    c[1] = 0
    c = 1/u * ( x - (1-exp(g*x))/(1-exp(g)) )
    g = u/D
    '''

    st = ScalarTransport(mesh_1d, 'a')
    st.set_element('CG', 1)
    st.build_function_space()
    u = 1
    h = 1/10 # mesh resolution
    Pe = 5 # This is cell Peclet number
    D = u/Pe/2*h
    r = 1.
    st.set_advection_velocity(u)
    st.set_diffusivity(D)
    st.set_reaction(r)

    # Set weak form
    st.set_weak_form()
    st.add_stab('su')
    bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.)},
               2:{'type': 'dirichlet', 'value': fe.Constant(0.)}}
    st.set_bcs(bc_dict)

    # Set solver
    la_solver = fe.PETScKrylovSolver()
    fe.PETScOptions.set("ksp_monitor")
    la_solver.set_from_options()
    solver = PhysicsSolver(st, la_solver=la_solver)

    # Solve
    solver.solve()

    # Compare solutions
    x = np.linspace(0, 1, 11)
    g = u/D
    sol_exact = 1/u * (x - (1-np.exp(g*x))/(1-np.exp(g)))
    sol_comp = [st.solution(fe.Point(xi)) for xi in x]

    # Since this is an approximation to the actual solution
    # Make sure that the error is less than a small number to the exact solution
    assert np.linalg.norm(sol_exact-sol_comp) < 1e-8


