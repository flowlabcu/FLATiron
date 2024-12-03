from flatiron_tk.physics import SteadyIncompressibleNavierStokes
from flatiron_tk.io import h5_mod
from flatiron_tk.mesh import Mesh, RectMesh
from flatiron_tk.solver import PhysicsSolver
import fenics as fe
from flatiron_tk.physics import StokesFlow

# Build mesh
ne = 64
mesh = RectMesh(0, 0, 1, 1, 1/ne)

# Define nse equation
nse = SteadyIncompressibleNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()

# Set parameters
Re = 100
mu = 1/Re
rho = 1
nse.set_density(rho)
nse.set_dynamic_viscosity(mu)

# Set weak form
nse.set_weak_form()
nse.add_stab()

# Boundary condition
zero_v = fe.Constant( (0,0) )
zero = fe.Constant(0)
u_bcs = {
        1: {'type': 'dirichlet', 'value': zero_v},
        2: {'type': 'dirichlet', 'value': zero_v},
        3: {'type': 'dirichlet', 'value': zero_v},
        4: {'type': 'dirichlet', 'value': fe.Constant((1, 0))},
        }
p_bcs = {'point_0': {'type': 'dirichlet', 'value': fe.Constant(0), 'x': (0, 0)}}
bc_dict = {'u': u_bcs,
           'p': p_bcs}
nse.set_bcs(bc_dict)

# Set output writer
nse.set_writer("output", "pvd")

# Solve and write result
solver = PhysicsSolver(nse)
solver.solve()
nse.write()

