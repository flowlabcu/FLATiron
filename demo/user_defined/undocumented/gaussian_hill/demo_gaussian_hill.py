'''
Demo for 1D transient convection-diffusion equation on an interval [0, 1]
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

This demo demonstrates how to do a transient convection-diffusion problem in flatiron_tk

Author: njrovito
'''

# ------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------- #

import fenics as fe
from flatiron_tk.physics import TransientScalarTransport
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import PhysicsSolver

# ------------------------------------------------------- #

# Define mesh
ne = 100                        # Number of points in mesh
IM = fe.IntervalMesh(ne, 0, 1)  # Initializing interval mesh object
mesh = Mesh(mesh=IM)            # Initializing flatiron mesh object
h = 1/float(ne)                 # Element size
a = 1                           # Controls CFL

# Mark mesh
def left(x, left_bnd):    # Defining left boundery function
    return abs(x[0] - left_bnd) < fe.DOLFIN_EPS # Epsilon - machine precision
def right(x, right_bnd):  # Defining right boundery function
    return abs(right_bnd - x[0]) < fe.DOLFIN_EPS

# Imposing bounderies on Mesh object
mesh.mark_boundary(1, left, (0.))
mesh.mark_boundary(2, right, (1.0))

# Define problem
C = 1.0     # CFL/Courant number
dt = C*h/a  # Time step
st = TransientScalarTransport(mesh, dt, theta=0.5)
st.set_element('CG', 1)
st.build_function_space()

# Diffusivity (here set as a constant)
Pe = 100
D = a*h/2/Pe
st.set_diffusivity(D, D)

# For the velocity term, we have a time-dependent velocity.
# We will create two separate functions u0  and un and update
# them with the appropriate t.
un = u0 = fe.Constant(a)
st.set_advection_velocity(u0, un)

# Similarly, we create f0 and fn for the reaction term (here set to zero)
st.set_reaction(0, 0)

# Set weak form
st.set_weak_form()

# Set initial condition
l = 7*np.sqrt(2)/300.
x0 = 2./15.
ic = fe.Expression("5./7. * exp(-(pow((x[0]-x0)/l, 2)))", degree=4, x0=x0, l=l)
ic = fe.interpolate(ic, st.V)
st.set_initial_condition(ic)

# Set bc
bc_dict = {1:{'type': 'dirichlet', 'value': fe.Constant(0.0)},
           2:{'type': 'dirichlet', 'value': fe.Constant(0.0)}}
st.set_bcs(bc_dict)

# Set problem
solver = PhysicsSolver(st)

# Begin transient section
t = 0.0
t_end = 0.6
while t <= t_end:
    t += dt
    solver.solve()
    st.update_previous_solution()

fe.plot(st.solution_function(), label='Computed solution:', color='blue')

# Plot computed solution against exact solution
x = np.linspace(0, 1, ne+1)
sigma = np.sqrt(1 + 4*D*t/l**2)
sol_exact = 5./(7*sigma) * np.exp(- ( (x-x0-a*t)/(l*sigma) )**2)
plt.plot(x, sol_exact, '--', color='black', label='Exact Solution')

plt.xlabel('x', fontsize=14)
plt.ylabel('c', rotation=0, fontsize=14)

plt.legend()
plt.grid(alpha=.3)

plt.title('t = %.4f' % t)
plt.ylim([-0.2, 0.8])
plt.show()




