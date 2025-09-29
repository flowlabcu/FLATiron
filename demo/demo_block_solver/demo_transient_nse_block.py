import matplotlib.pyplot as plt
import numpy as np

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()
MPI = import_mpi4py()

from flatiron_tk.mesh import Mesh
from flatiron_tk.physics import TransientNavierStokes
from flatiron_tk.solver import BlockNonLinearSolver
from flatiron_tk.solver import BlockSplitTree
from flatiron_tk.solver import ConvergenceMonitor
from flatiron_tk.solver import NonLinearProblem


# Define boundary condition functions
def no_slip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def zero_pressure(x):
    return np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type)

def inlet_velocity(x, t):
    values = np.zeros((2, x.shape[1]), dtype=dolfinx.default_scalar_type)
    values[0] = np.sin(t)
    return values

# Create the Mesh object by reading the mesh from file
mesh_file = '../mesh/bfs.msh'
mesh = Mesh(mesh_file=mesh_file)

# Create the TransientNavierStokes object
nse = TransientNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()

# Set physical parameters
dt = 0.05
mu = 0.001
rho = 1
nse.set_time_step_size(dt)
nse.set_midpoint_theta(0.5)
nse.set_density(rho)
nse.set_dynamic_viscosity(mu)

# Add stabilization
nse.set_weak_form(stab=True)

# Get function spaces to build boundary conditions
V_u = nse.get_function_space('u').collapse()[0]
V_p = nse.get_function_space('p').collapse()[0]

# Create functions for boundary conditions
inlet_v = dolfinx.fem.Function(V_u)

# At each time step:
t = 0.0
inlet_v.interpolate(lambda x: inlet_velocity(x, t))

zero_p = dolfinx.fem.Function(V_p)
zero_p.interpolate(zero_pressure)

zero_v = dolfinx.fem.Function(V_u)
zero_v.interpolate(no_slip)

# Define boundary conditions dictionary
u_bcs = {
        7: {'type': 'dirichlet', 'value': inlet_v},
        8: {'type': 'dirichlet', 'value': zero_v},
        10: {'type': 'dirichlet', 'value': zero_v},
        }

p_bcs = {
        9: {'type': 'dirichlet', 'value': zero_p},
        }

bc_dict = {'u': u_bcs, 
           'p': p_bcs}

nse.set_bcs(bc_dict)

# Set the output writer
nse.set_writer('output', 'xdmf')

# Define the problem
problem = NonLinearProblem(nse)

# Set up Block Solver 
# U block preconditioner parameters
def set_ksp_u(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setMonitor(ConvergenceMonitor("|----KSPU", verbose=True))
    ksp.setTolerances(max_it=3)
    ksp.pc.setType(PETSc.PC.Type.JACOBI)
    ksp.setUp()

# P block preconditioner parameters
def set_ksp_p(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setMonitor(ConvergenceMonitor("|--------KSPP", verbose=True))
    ksp.setTolerances(max_it=5)
    ksp.pc.setType(PETSc.PC.Type.HYPRE)
    ksp.pc.setHYPREType("boomeramg")
    ksp.setUp()

# Outer solver parameters
def set_outer_ksp(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setGMRESRestart(30)
    ksp.setTolerances(rtol=1e-100, atol=1e-10)
    ksp.setMonitor(ConvergenceMonitor("Outer ksp"))

# Define the block structure 
split = {
        'fields': ('u', 'p'),
        'composite_type': 'schur',
        'schur_fact_type': 'full',
        'schur_pre_type': 'a11',
        'ksp0_set_function': set_ksp_u,
        'ksp1_set_function': set_ksp_p
        }

# Create the Block Tree structure
tree = BlockSplitTree(nse, splits=split)

# Create the Block Nonlinear Solver
solver = BlockNonLinearSolver(tree, MPI.COMM_WORLD, problem, outer_ksp_set_function=set_outer_ksp)

while t < 10.0:
    print(f'Solving at time t = {t:.2f}')
    
    # Set the inlet velocity for the current time step
    inlet_v.interpolate(lambda x: inlet_velocity(x, t))
    

    # Solve the problem
    solver.solve()

    nse.update_previous_solution()
    nse.write(time_stamp=t)
    
    # Update time
    t += dt
