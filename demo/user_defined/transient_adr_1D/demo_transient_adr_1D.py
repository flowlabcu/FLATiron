'''
Demo for 1D transient convection-diffusion equation on an interval [0,12800]
with no reactions
dc/dt = D*d^2c/dx^2 - u*dc/dx - f

The following problem was taken from "Problem 1" from "Benchmarks for the Transport Equation:
The Convection_Diffusion Forum and Beyond" by Baptista and Adams, 1995

D = 2
f = 0
u = 1.5*sin(2*pi*t/9600)
The Gaussian Source Solution is:
c(x,t) = sigma_0/sigma * exp(-(x-x_bar)^2 / 2*sigma^2)
sigma^2 = sigma_0^2 + 2*D*t
x_bar = x_0 + int(u(T)dT) from 0 to T

This demo demonstrates how to do a transient convection-diffusion problem in feFlow

Author: njrovito
'''

# ------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------- #

import fenics as fe
from feFlow.physics import TransientScalarTransport
from feFlow.mesh import Mesh
from feFlow.solver import PhysicsSolver

# ------------------------------------------------------- #

# Define mesh
ne = 128
IM = fe.IntervalMesh(ne,0,12800)
mesh = Mesh(mesh=IM)

# Mark mesh
def left(x, left_bnd):
    return abs(x[0] - left_bnd) < fe.DOLFIN_EPS
def right(x, right_bnd):
    return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
mesh.mark_boundary(1,left,(0.))
mesh.mark_boundary(2, right, (12800))

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

# Define problem
dt = 96
t_0 = 3000
st = TransientScalarTransport(mesh, dt, theta=0.5)
st.set_element('CG', 1)
st.build_function_space()

# Diffusivity (here set as a constant)
D = 2
st.set_diffusivity(D, D)

# For the velocity term, we have a time-dependent velocity.
# We will create two separate functions u0  and un and update
# them with the appropriate t.
a = 1.5
b = 2 * np.pi / 9600
u0 = fe.Expression('a * sin(b*t)', degree=1, a=a, b=b, t=0)
un = fe.Expression('a * sin(b*t)', degree=1, a=a, b=b, t=0)
st.set_advection_velocity(u0, un)

# Similarly, we create f0 and fn for the reaction term (here set to zero)
st.set_reaction(0, 0)

# Set weak form
st.set_weak_form()

# Set initial condition
x = np.linspace(0, 12800, ne, endpoint=True)
sigma_0 = 264
sigma = get_sigma(sigma_0, D, t_0)
x_bar = get_x_bar(a, b, t_0)
c0 = fe.interpolate(fe.Expression('s_0/s * exp(-1*pow(x[0]-x_bar,2)/(2*pow(s,2)))',
                                  s_0=sigma_0, s=sigma, x_bar=x_bar, degree=1), st.V)
st.set_initial_condition(c0)

# Set bc
bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.)},
           2:{'type': 'dirichlet', 'value': fe.Constant(0.)}}
st.set_bcs(bc_dict)

# Set problem
solver = PhysicsSolver(st)

# Begin transient section
t = t_0
t_end = 7200
while t <= t_end:

    # Set velocity at current and previous step
    u0.t = t
    un.t = t + dt

    # Solve
    st.set_advection_velocity(u0, un)
    solver.solve()

    # Update previous solution
    st.update_previous_solution()

    # Update time
    t += dt

    # Plot computed solution against exact solution
    sol_exact = get_c_exact(x, a, b, t, D, sigma_0)
    fe.plot(st.solution_function(), label='Computed solution')
    plt.plot(x, sol_exact, 'r--', label='Exact solution')
    plt.legend()
    plt.title('t = %.4f' % t)
    plt.ylim([-0.1, 1.1])
    plt.pause(0.01)
    plt.cla()


