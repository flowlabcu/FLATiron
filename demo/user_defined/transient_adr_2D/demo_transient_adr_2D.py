'''
This demo demonstrates the 2D transient transport problem in flatiron_tk
Here, we use the analytical results found in: DOI:10.1029/CE047p0241

Problem 4: Diffusion in a plane shear flow.

The problem description is as follows:

    Strong form:
        \partial_t c + (u_0 + \lambda y) \partial_x c = D \nabla^2 c
        c(x,y,t=0) = M*DiracDelta(x0, 0) # (DiracDelta(x,y))
    
    Domain: 
        x, y -> infty
        To approximate this, we use a big enough domain
        and set a zero flux condition at all of the boundaries

    Exact solution:
        c(x, y, t) = M/k1 * Exp( - (k2/k3) + (k4/k5) )
        k1 = 4*pi*D*t*(1 + \lambda**2 * t**2 / 12)
        k2 = ( x - xbar - 0.5*\lambda*y*t )**(1/2)
        k3 = 4*D*t*(1 + \lambda**2*t**2/12)
        k4 = y**2
        k5 = 4*D*t
        xbar = x0 + u0*t (typo in the paper)

        where M, x0, u0, \lambda, D are constants

'''

import numpy as np
import matplotlib.pyplot as plt
import sys

# ------------------------------------------------------- #

import fenics as fe
from flatiron_tk.physics import TransientScalarTransport
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import PhysicsSolver


# Define problem definitions
u0 = 0.5
t0 = 2400
x0 = 7200
D = 10
lam = 5e-4
dt = 96
t_end = 9600
M = 1

const = {
        'u0': u0,
        'x0': x0,
        'D': D,
        'lam': lam,
        'M': M
        }

c_exct = fe.Expression( (" M/( 4*pi*D*t*pow(1 + pow(lam,2)*pow(t,2)/12., 0.5) ) * exp(- (  pow( x[0] - (x0+u0*t) - 0.5*lam*x[1]*t, 2 )/( 4*D*t*(1 + pow(lam, 2)*pow(t, 2)/12) ) + (pow(x[1],2))/(4*D*t)  )  ) "), degree=4, t=t0, **const)

# Define mesh
m_x0 = 0
m_y0 = -3400
m_x1 = 24000
m_y1 = 3400
nex = 300
ney = int(nex/((m_x1-m_x0)/(m_y1-m_y0)))
RM = fe.RectangleMesh(fe.Point(m_x0, m_y0), fe.Point(m_x1, m_y1), nex, ney, 'crossed')
mesh = Mesh(mesh=RM)

# Mark mesh
def left(x, left_bnd):
    return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
def right(x, right_bnd):
    return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
def top(x, top_bnd):
    return abs(x[1] - top_bnd) < fe.DOLFIN_EPS
def bottom(x, bottom_bnd):
    return abs(bottom_bnd - x[1]) < fe.DOLFIN_EPS
mesh.mark_boundary(1, left, (m_x0))
mesh.mark_boundary(2, bottom, (m_y0))
mesh.mark_boundary(3, right, (m_x1))
mesh.mark_boundary(4, top, (m_y1))

# Define problem
physics = TransientScalarTransport(mesh, dt, theta=0.5)
physics.set_element('CG', 1)
physics.build_function_space()

# Set coefficients on each term
# Since D and u are constant in time, we repeat the values for both entry
# Zero reaction
u = fe.Expression(("u0 + lam*x[1]", "0."), degree=1, u0=u0, lam=lam)
physics.set_advection_velocity(u, u)
physics.set_diffusivity(D, D)
physics.set_reaction(0, 0)

# Set weak form
physics.set_weak_form(stab=True)

# Set initial condition
c_exct.t = t0
c0 = fe.interpolate(c_exct, physics.V)
physics.set_initial_condition(c0)

# Set bc
# The domain goes to infinity
# for a finite domain, we use a zero-flux condition
bc_dict = {}
physics.set_bcs(bc_dict)

# Set problem
solver = PhysicsSolver(physics)

# Solve the problem
err = fe.Function(physics.V)
computed_results = []
i = 0
t = t0
physics.set_writer("output", 'pvd')
fid_exct = fe.File("output/exact.pvd")
error_over_time = []
while t <= t_end:

    # Update time
    t = t + dt
    i += 1

    # Solve
    solver.solve()
    physics.write(t=t)

    # Update previous solution
    physics.update_previous_solution()

    # Compute exact solution
    c_exct.t = t
    c_exact = fe.interpolate(c_exct, physics.V)
    c_exact.rename("c_exact", "c_exact")
    fid_exct << c_exact

    # Compute L2 error norm
    err.vector()[:] = c_exact.vector()[:] - physics.solution_function().vector()[:]
    error_over_time.append(fe.norm(err.vector()))
    plt.semilogy(error_over_time)
    plt.title("L2 error over time")
    plt.ylabel('L2 error over time')
    plt.xlabel('Time step')
    plt.pause(0.001)
    plt.cla()


