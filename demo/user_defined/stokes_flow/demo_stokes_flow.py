import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------- #

import fenics as fe
from flatiron_tk.physics import StokesFlow
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import PhysicsSolver
import sys

# Define mesh
ne = 64
RM = fe.UnitSquareMesh(ne, ne)
mesh = Mesh(mesh=RM)

# Mark mesh
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

# Define problem
ics = StokesFlow(mesh)
ics.set_element('CG', 1, 'CG', 1)
ics.build_function_space()

# Set coefficients on each term
# here since we are in transient mode, we have to set
# the function defining the previous and current time step.
# Since D and u are constants, we repeat the values for both entry
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
ics.set_body_force(b)
nu = 1.
ics.set_kinematic_viscosity(nu)

# Set weak form
ics.set_weak_form()

# Set stabilization
ics.add_stab()

# Set bc
u_bcs = {
        1: {'type': 'dirichlet', 'value': 'zero'},
        2: {'type': 'dirichlet', 'value': 'zero'},
        3: {'type': 'dirichlet', 'value': 'zero'},
        4: {'type': 'dirichlet', 'value': 'zero'}
        }
p_bcs = {'point_0': {'type': 'dirichlet', 'value':'zero', 'x': (0., 0.)}}
bc_dict = {'u': u_bcs, 
           'p': p_bcs}
ics.set_bcs(bc_dict)

# Manually add a point-wise boundary condition for pressure to set pressure level
# TODO: Integrate this into flatiron_tk
# def bottom_left_corner(x, on_boundary):
#     return fe.near(x[0], 0.) and fe.near(x[1], 0.)
# ics.dirichlet_bcs.append(fe.DirichletBC(ics.V.sub(1), fe.Constant(0), bottom_left_corner, method='pointwise'))

# Setup io
ics.set_writer('output', 'h5')

# Set problem
la_solver = fe.LUSolver()
solver = PhysicsSolver(ics, la_solver)

# Solve
solver.solve()

u0e = " pow(x[0], 2)*pow(1-x[0], 2)*(2*x[1] - 6*pow(x[1], 2) + 4*pow(x[1], 3))"
u1e = "-pow(x[1], 2)*pow(1-x[1], 2)*(2*x[0] - 6*pow(x[0], 2) + 4*pow(x[0], 3))"
u_exact = fe.Expression( (u0e, u1e), degree=2 )
u_exact = fe.interpolate(u_exact, ics.V.sub(0).collapse())
p_exact = fe.Expression("x[1]*(1-x[1])", degree=2)
p_exact = fe.interpolate(p_exact, ics.V.sub(1).collapse())
fe.File("output/ue.pvd") << u_exact
ics.write()


# Plot solution against exact solution
n_sample = 50
span = np.linspace(0, 1, n_sample)

u_exact_vals = []
u_comp_vals = []
p_exact_vals = []
p_comp_vals = []
for xi, yi in zip(span, span):
    u, p = ics.solution_function().split(deepcopy=True)

    u_exact_vals.append(u_exact(fe.Point(xi, yi)))
    u_comp_vals.append(u(fe.Point(xi, yi)))

    p_exact_vals.append(p_exact(fe.Point(xi, yi)))
    p_comp_vals.append(p(fe.Point(xi, yi)))



fig, ax = plt.subplots(nrows = 3)
u_exact_vals = np.array(u_exact_vals)
u_comp_vals = np.array(u_comp_vals)
p_exact_vals = np.array(p_exact_vals)
p_comp_vals = np.array(p_comp_vals)

fig.suptitle('Values across diagonal')
ax[0].plot(u_comp_vals[:,0], label='Computed')
ax[0].plot(u_exact_vals[:,0], '--', label='Exact')
ax[0].set_title('x velocity')
ax[0].grid(True)
ax[0].legend()

ax[1].plot(u_comp_vals[:,1], label='Computed')
ax[1].plot(u_exact_vals[:,1], '--', label='Exact')
ax[1].set_title('y velocity')
ax[1].grid(True)
ax[1].legend()

ax[2].plot(p_comp_vals, label='Computed')
ax[2].plot(p_exact_vals, '--', label='Exact')
ax[2].set_title('Pressure')
ax[2].grid(True)
ax[2].legend()

plt.show()




