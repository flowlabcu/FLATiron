import fenics as fe 
import numpy as np 
import pytest 
from flatiron_tk.physics import *
from flatiron_tk.solver import PhysicsSolver

def test_scalar_transport(mesh_1d):
    '''
    This test solves the scalar tranport problem found in the demo.
    dc/dt = D*d^2c/dx^2 - u*dc/dx - f
    D = 2
    f = 0
    u = 1.5*sin(2*pi*t/9600)

    c(x,t) = sigma_0/sigma * exp(-(x-x_bar)^2 / 2*sigma^2)
    sigma^2 = sigma_0^2 + 2*D*t
    x_bar = x_0 + int(u(T)dT) from 0 to T

    Author: njrovito
    '''
    # Defines x_bar
    def get_x_bar(a_a,a_b,a_t):
        x_bar = (a_a - a_a * np.cos(a_b * a_t)) / a_b
        return x_bar

    # Defines sigma
    def get_sigma(a_sigma_0,a_D,a_t):
        sigma = np.sqrt(a_sigma_0**2 + 2 * a_D * a_t)
        return sigma

    # Defines exact solution
    def get_c_exact(a_x, a_a,a_b,a_t,a_D,a_sigma_0):
        sigma = get_sigma(a_sigma_0, a_D, a_t)
        x_bar = get_x_bar(a_a, a_b, a_t)
        c = (a_sigma_0 / sigma) * np.exp(-1 * (a_x - x_bar)**2 / (2 * sigma**2))
        return c

    # Make mesh
    mesh = mesh_1d
    def left(x, left_bnd):
        return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
    def right(x, right_bnd):
        return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
    mesh.mark_boundary(1,left,(0.))
    mesh.mark_boundary(2, right, (12800))

    dt = 96
    t_0 = 3000
    D = 2

    stp = TransientScalarTransport(mesh_1d, dt, theta=0.5)
    stp.set_element('CG', 1)
    stp.build_function_space()
    stp.set_diffusivity(D, D)

    a = 1.5
    b = 2 * np.pi / 9600
    u0 = fe.Expression('a * sin(b*t)', degree=1, a=a, b=b, t=0)
    un = fe.Expression('a * sin(b*t)', degree=1, a=a, b=b, t=0)
    stp.set_advection_velocity(u0, un)

    stp.set_reaction(0, 0)
    stp.set_weak_form()

    # Set initial condition
    x = np.linspace(0, 1, 11)
    sigma_0 = 264
    sigma = get_sigma(sigma_0, D, t_0)
    x_bar = get_x_bar(a, b, t_0)
    c0 = fe.interpolate(fe.Expression('s_0/s * exp(-1*pow(x[0]-x_bar,2)/(2*pow(s,2)))',
                                    s_0=sigma_0, s=sigma, x_bar=x_bar, degree=1), stp.V)
    stp.set_initial_condition(c0)

    # Set bc
    bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.)},
            2:{'type': 'dirichlet', 'value': fe.Constant(0.)}}
    stp.set_bcs(bc_dict)

    # Set problem
    solver = PhysicsSolver(stp)

    t = t_0
    t_end = 7200
    err = []
    while t <= t_end:

        # Set velocity at current and previous step
        u0.t = t
        un.t = t + dt

        # Solve
        stp.set_advection_velocity(u0, un)
        solver.solve()

        # Update previous solution
        stp.update_previous_solution()

        # Update time
        t += dt

        # compare solutions
        sol_comp = [stp.solution(fe.Point(xi)) for xi in x]
        sol_exact = get_c_exact(x, a, b, t, D, sigma_0)

        err.append(np.linalg.norm(sol_exact-sol_comp)/11)

    assert(np.max(err)< 1e-10) # loose tolerances for coarse mesh



