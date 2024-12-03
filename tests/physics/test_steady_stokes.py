import fenics as fe
import numpy as np
import pytest
from flatiron_tk.physics import StokesFlow
from flatiron_tk.solver import PhysicsSolver

def test_steady_stokes_velocity_driven(mesh_2d):
    '''
    This test solves the problem found in the demo
    demo_stokes_flow.py

    The exact solution is:
    u =  pow(x[0], 2)*pow(1-x[0], 2)*(2*x[1] - 6*pow(x[1], 2) + 4*pow(x[1], 3))
    v = -pow(x[1], 2)*pow(1-x[1], 2)*(2*x[0] - 6*pow(x[0], 2) + 4*pow(x[0], 3))

    Author: njrovito
    '''
    # Meshing
    mesh = mesh_2d
    def left(x, left_bnd):
        return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
    def right(x, right_bnd):
        return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
    def top(x, top_bnd):
        return abs(top_bnd - x[1]) < fe.DOLFIN_EPS
    def bottom(x, bottom_bnd):
        return abs(x[1] - bottom_bnd) < fe.DOLFIN_EPS
    mesh.mark_boundary(1, left, (0.))
    mesh.mark_boundary(2, bottom, (0.))
    mesh.mark_boundary(3, right, (1.))
    mesh.mark_boundary(4, top, (1.))

    stk = StokesFlow(mesh)
    stk.set_element('CG', 2, 'CG', 1)
    stk.build_function_space()

    bx4 = "( 12 - 24*x[1]) * pow(x[0], 4)"
    bx3 = "(-24 + 48*x[1]) * pow(x[0], 3)"
    bx2 = "( 12 - 48*x[1] + 72*pow(x[1], 2) - 48*pow(x[1], 3) ) * pow(x[0], 2)"
    bx1 = "( -2 + 24*x[1] - 72*pow(x[1], 2) + 48*pow(x[1], 3) ) * x[0]"
    bx0 = "(  1 -  4*x[1] + 12*pow(x[1], 2) -  8*pow(x[1], 3) )"

    by3 = "(  8 - 48*x[1] + 48*pow(x[1], 2) ) * pow(x[0], 3)"
    by2 = "(-12 + 72*x[1] - 72*pow(x[1], 2) ) * pow(x[0], 2)"
    by1 = "(  4 - 24*x[1] + 48*pow(x[1], 2) - 48*pow(x[1], 3) + 24*pow(x[1], 4) ) * x[0]"
    by0 = "(              - 12*pow(x[1], 2) + 24*pow(x[1], 3) - 12*pow(x[1], 4) )"
    bx = "%s + %s + %s + %s + %s" % (bx4, bx3, bx2, bx1, bx0)
    by = "%s + %s + %s + %s" % (by3, by2, by1, by0)
    b = fe.Expression((bx, by), degree=4)
    stk.set_body_force(b)
    nu = 1.
    stk.set_kinematic_viscosity(nu)
    stk.set_weak_form()

    u_bcs = {
        1: {'type': 'dirichlet', 'value': 'zero'},
        2: {'type': 'dirichlet', 'value': 'zero'},
        3: {'type': 'dirichlet', 'value': 'zero'},
        4: {'type': 'dirichlet', 'value': 'zero'}
        }
    p_bcs = {'point_0': {'type': 'dirichlet', 'value':'zero', 'x': (0., 0.)}}
    bc_dict = {'u': u_bcs, 
            'p': p_bcs}
    stk.set_bcs(bc_dict)

    la_solver = fe.LUSolver()
    solver = PhysicsSolver(stk, la_solver)

    # Solve
    solver.solve()
    u0e = " pow(x[0], 2)*pow(1-x[0], 2)*(2*x[1] - 6*pow(x[1], 2) + 4*pow(x[1], 3))"
    u1e = "-pow(x[1], 2)*pow(1-x[1], 2)*(2*x[0] - 6*pow(x[0], 2) + 4*pow(x[0], 3))"
    u_exact = fe.Expression( (u0e, u1e), degree=2 )
    u_exact = fe.interpolate(u_exact, stk.V.sub(0).collapse())
    p_exact = fe.Expression("x[1]*(1-x[1])", degree=2)
    p_exact = fe.interpolate(p_exact, stk.V.sub(1).collapse())

    span = np.linspace(0, 1, 11)
    u_exact_vals = []
    u_comp_vals = []
    p_exact_vals = []
    p_comp_vals = []
    for xi, yi in zip(span, span):
        u, p = stk.solution_function().split(deepcopy=True)

        u_exact_vals.append(u_exact(fe.Point(xi, yi)))
        u_comp_vals.append(u(fe.Point(xi, yi)))

        p_exact_vals.append(p_exact(fe.Point(xi, yi)))
        p_comp_vals.append(p(fe.Point(xi, yi)))

    # NOTE: we're using a coarse mesh, so we use a loose tolerance
    assert np.linalg.norm(np.array(u_exact_vals) - np.array(u_comp_vals)) < 1e-4 # Computed velocity does not match exact solution
    assert np.linalg.norm(np.array(p_exact_vals) - np.array(p_comp_vals)) < 1e-4 # Computed pressure does not match exact solution

    

