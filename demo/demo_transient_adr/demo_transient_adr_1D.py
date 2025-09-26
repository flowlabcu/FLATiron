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

This demo demonstrates how to do a transient convection-diffusion problem in flatiron_tk

Author: njrovito
'''
import dolfinx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import ufl

from flatiron_tk.mesh import LineMesh
from flatiron_tk.physics import TransientScalarTransport
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver

# Functions for the exact solution
def get_x_bar(a_a, a_b, a_t):
    x_bar = (a_a - a_a * np.cos(a_b * a_t)) / a_b
    return x_bar

def get_sigma(a_sigma_0, a_D, a_t):
    sigma = np.sqrt(a_sigma_0**2 + 2 * a_D * a_t)
    return sigma

def get_c_exact(a_x, a_a, a_b, a_t, a_D, a_sigma_0):
    sigma = get_sigma(a_sigma_0, a_D, a_t)
    x_bar = get_x_bar(a_a, a_b, a_t)
    c = (a_sigma_0 / sigma) * np.exp(-1 * (a_x - x_bar)**2 / (2 * sigma**2))
    return c

# Create a Line Mesh 
num_elements = 128 
h = 12800 / num_elements
mesh = LineMesh(0, 12800, h)

# Define problem
dt = 96
a = 1.5
b = 2 * np.pi / 9600
time = dolfinx.fem.Constant(mesh.msh, 0.0)

stp = TransientScalarTransport(mesh, dt, tag='c')
stp.set_element('CG', 1)
stp.build_function_space()
V = stp.get_function_space()

# Set diffusivity 
diffusivity = 2.0
stp.set_diffusivity(diffusivity, diffusivity)

# Create a function for the advection velocity
u0 = dolfinx.fem.Function(V)
un = dolfinx.fem.Function(V)
u0.name = 'u0'
un.name = 'un'

# Interpoate a ufl expression for the advection velocity
u_expr = a * ufl.sin(b * time)
interpolation_points = V.element.interpolation_points()
u0.interpolate(dolfinx.fem.Expression(u_expr, interpolation_points))
un.interpolate(dolfinx.fem.Expression(u_expr, interpolation_points))
stp.set_advection_velocity(u0, un)

# Set reaction term
stp.set_reaction(0.0, 0.0)

# Set weak form and stabilization
stp.set_weak_form()
stp.add_stab()

# Set intial condition
x = ufl.SpatialCoordinate(mesh.msh)
t_0 = 1000
sigma_0 = 264
sigma = get_sigma(sigma_0, diffusivity, t_0)
x_bar = get_x_bar(a, b, t_0)
c0 = dolfinx.fem.Function(V)
c0_expr = (sigma_0 / sigma) * ufl.exp(-(x[0] - x_bar)**2 / (2 * sigma**2))
c0.interpolate(dolfinx.fem.Expression(c0_expr, interpolation_points))
stp.set_initial_condition(c0)

# Set boundary conditions
bc_dict = {1: {'type': 'dirichlet', 'value': dolfinx.fem.Constant(mesh.msh, 0.0)},
           2: {'type': 'dirichlet', 'value': dolfinx.fem.Constant(mesh.msh, 0.0)}}
stp.set_bcs(bc_dict)

# Set up the solver
problem = NonLinearProblem(stp)
solver = NonLinearSolver(mesh.msh.comm, problem)
stp.set_writer('output', 'xdmf')

# Set up the time-stepping
t = t_0
u_vals = []
t_vals = []

# Create an array for the x values to plot the solutions against
x_plt = np.linspace(0, 12800, num_elements + 1, endpoint=True)
# Create lists to hold the numerical and exact solutions at each time step
c_vals_list = []
sol_exact_list = []
time_vals = []

while t < 8000:
    # Update advection velocity
    time.value = t # Update ufl time expression
    u0.interpolate(dolfinx.fem.Expression(a * ufl.sin(b * time), interpolation_points))
    time.value = t + dt
    un.interpolate(dolfinx.fem.Expression(a * ufl.sin(b * time), interpolation_points))

    # Solve the current time step
    solver.solve()
    stp.update_previous_solution()

    # Write the solution 
    stp.write(time_stamp=t)

    # Plot the numerical and exact solutions
    sol_exact = get_c_exact(x_plt, a, b, t, diffusivity, sigma_0)
    c_vals = stp.get_solution_function().x.array

    c_vals = stp.get_solution_function().x.array.copy()
    sol_exact = get_c_exact(x_plt, a, b, t, diffusivity, sigma_0)

    c_vals_list.append(c_vals)
    sol_exact_list.append(sol_exact)
    time_vals.append(t)

    # Step forward in time
    t += dt


# Plot the evolution of the solution over time using matplotlib animation
fig, ax = plt.subplots()
line_num, = ax.plot([], [], label='Numerical')
line_exact, = ax.plot([], [], label='Exact')
ax.set_xlim(0, 12800)
ax.set_ylim(-0.1, 1.1)
ax.legend()

def update(frame):
    line_num.set_data(x_plt, c_vals_list[frame])
    line_exact.set_data(x_plt, sol_exact_list[frame])
    ax.set_title(f"t = {time_vals[frame]:.0f}")
    return line_num, line_exact

ani = animation.FuncAnimation(fig, update, frames=len(time_vals), blit=True, interval=100)
ani.save('solution_evolution.mp4', writer='ffmpeg')
plt.close(fig)

