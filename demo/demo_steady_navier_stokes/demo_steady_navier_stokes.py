import dolfinx
import numpy as np

from flatiron_tk.mesh import RectMesh
from flatiron_tk.physics import SteadyNavierStokes
from flatiron_tk.solver  import ConvergenceMonitor
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver

# Define boundary conditions functions
def no_slip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def u_inlet(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

def zero_pressure(x):
    return np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type)

# Create mesh
ne = 64
mesh = RectMesh(0.0, 0.0, 1.0, 1.0, 1/ne)

# Build Navier-Stokes problem
nse = SteadyNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()

# Set physical parameters
Re = 100.0
mu = 1.0 / Re
rho = 1.0
nse.set_density(rho)
nse.set_dynamic_viscosity(mu)

# Set weak form and stabilization
nse.set_weak_form()
nse.add_stab()

# Velocity and pressure subspaces
V_u = nse.get_function_space('u').collapse()[0]
V_p = nse.get_function_space('p').collapse()[0]

# Create boundary condition functions
zero_v = dolfinx.fem.Function(V_u)
zero_v.interpolate(no_slip)
inlet_v = dolfinx.fem.Function(V_u)
inlet_v.interpolate(u_inlet)
zero_p = dolfinx.fem.Function(V_p)
zero_p.x.array[:] = 0.0

# Define boundary conditions
u_bcs = {
        1: {'type': 'dirichlet', 'value': zero_v},
        2: {'type': 'dirichlet', 'value': zero_v},
        3: {'type': 'dirichlet', 'value': zero_v},
        4: {'type': 'dirichlet', 'value': inlet_v},
        }

p_bcs = {
        1: {'type': 'dirichlet', 'value': zero_p},
        }

bc_dict = {'u': u_bcs, 
           'p': p_bcs}

nse.set_bcs(bc_dict)

nse.set_writer('output', 'pvd')

# Define problem
problem = NonLinearProblem(nse)

# Custom KSP setup function
def my_custom_ksp_setup(ksp):
    ksp.setType(ksp.Type.FGMRES)        
    ksp.pc.setType(ksp.pc.Type.LU)  
    ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)
    ksp.setMonitor(ConvergenceMonitor('ksp'))

# Create nonlinear solver
solver = NonLinearSolver(mesh.msh.comm, problem, outer_ksp_set_function=my_custom_ksp_setup)

# Solve the problem
solver.solve()
nse.write()