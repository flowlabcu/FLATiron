'''
This demo demonstrates the 2D transient transport problem in flatiron_tk
Here, we use the analytical results found in: DOI:10.1029/CE047p0241

Problem 4: Diffusion in a plane shear flow.

The problem description is as follows:

    Strong form:
        \\partial_t c + (u_0 + \\lambda y) \\partial_x c = D \\nabla^2 c
        c(x,y,t=0) = M*DiracDelta(x0, 0) # (DiracDelta(x,y))
    
    Domain: 
        x, y -> infty
        To approximate this, we use a big enough domain
        and set a zero flux condition at all of the boundaries

    Exact solution:
        c(x, y, t) = M/k1 * Exp( - (k2/k3) + (k4/k5) )
        k1 = 4*pi*D*t*(1 + \\lambda**2 * t**2 / 12)
        k2 = ( x - xbar - 0.5*\\lambda*y*t )**(1/2)
        k3 = 4*D*t*(1 + \\lambda**2*t**2/12)
        k4 = y**2
        k5 = 4*D*t
        xbar = x0 + u0*t (typo in the paper)

        where M, x0, u0, \\lambda, D are constants

'''

import numpy as np
import matplotlib.pyplot as plt

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()
MPI = import_mpi4py()

from flatiron_tk.physics import TransientScalarTransport
from flatiron_tk.mesh import RectMesh
from flatiron_tk.solver import NonLinearProblem, NonLinearSolver

# Define problem definitions
u0 = 0.5
t0 = 2400.0
x0 = 7200.0
D = 10.0
lam = 5e-4
dt = 96.0
t_end = 9600.0
M = 1.0

x0 = 0.0 
x1 = 24000.0
y0 = -3400.0
y1 = 3400.0

dx = (x1 - x0) / 100
dy = (y1 - y0) / 100

mesh = RectMesh(x0, y0, x1, y1, [dx, dy])

dt = 96
theta = 0.5
stp = TransientScalarTransport(mesh, dt, theta)
stp.set_element('CG', 1)
stp.build_function_space() 

# Define advection velocity
QE = basix.ufl.element('CG', mesh.msh.basix_cell(), 1, shape=(mesh.get_tdim(),))
W = dolfinx.fem.functionspace(mesh.msh, QE)
u = dolfinx.fem.Function(W)
x = ufl.SpatialCoordinate(mesh.msh)
u_expr = ufl.as_vector([u0 + lam * x[1], 0.0])
u.interpolate(dolfinx.fem.Expression(u_expr, W.element.interpolation_points()))

# Set the advection velocity, diffusivity, and reaction
stp.set_advection_velocity(u, u)
stp.set_diffusivity(D, D)
stp.set_reaction(0.0, 0.0)

# Build the weak form
stp.set_weak_form()
stp.add_stab()

# Set initial condition
V = stp.get_function_space()
c_exact = dolfinx.fem.Function(V)
c_exact.name = 'c_exact'
time = dolfinx.fem.Constant(mesh.msh, t0)

# Constants in expression
x0_const = dolfinx.fem.Constant(mesh.msh, x0)
u0_const = dolfinx.fem.Constant(mesh.msh, u0)
lam_const = dolfinx.fem.Constant(mesh.msh, lam)
D_const = dolfinx.fem.Constant(mesh.msh, D)
M_const = dolfinx.fem.Constant(mesh.msh, M)

# Expression for the exact solution
denom = 4 * D_const *time* ufl.sqrt(1 + (lam_const**2 * time**2) / 12)
gaussian_x = ((x[0] - (x0_const + u0_const * time) - 0.5 * lam_const * x[1] * time)**2) / (
    4 * D_const *time* (1 + (lam_const**2 * time**2) / 12))
gaussian_y = (x[1]**2) / (4 * D_const * time)
prefactor = M_const / (4 * np.pi * D_const *time* ufl.sqrt(1 + (lam_const**2 * time**2) / 12))

c_expr = prefactor * ufl.exp(- (gaussian_x + gaussian_y))

# Interpolate into a Function
c_exact.interpolate(dolfinx.fem.Expression(c_expr, V.element.interpolation_points()))
stp.set_initial_condition(c_exact)

# Boundary conditions
bc_dict = {}
stp.set_bcs(bc_dict)

# Set the problem and solver
problem = NonLinearProblem(stp)
solver = NonLinearSolver(mesh.msh.comm, problem)
stp.set_writer('output', 'pvd')

# set the exact writer 
exact_file = dolfinx.io.VTKFile(mesh.msh.comm, 'output/c_exact.pvd', 'w')


t = t0
error_over_time = []
err = dolfinx.fem.Function(V)
while t < t_end:
    solver.solve()
    stp.write(time_stamp=t)
    stp.update_previous_solution()

    time.value = t
    c_exact.interpolate(dolfinx.fem.Expression(c_expr, V.element.interpolation_points()))
    exact_file.write_function(c_exact)

    err.x.array[:] = c_exact.x.array[:] - stp.get_solution_function().x.array[:]
    l2_error = np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(err, err) * ufl.dx)))
    error_over_time.append(l2_error)

    # Plotting
    plt.semilogy(error_over_time)
    plt.title("L2 error over time")
    plt.ylabel("L2 error")
    plt.xlabel("Time step")
    plt.pause(0.001)
    plt.cla()

    t += dt

