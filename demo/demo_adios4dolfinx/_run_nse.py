import numpy as np
import matplotlib.pyplot as plt
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver
import dolfinx
import ufl 
import numpy as np
from flatiron_tk.physics import TransientNavierStokes
from flatiron_tk.solver  import ConvergenceMonitor
from flatiron_tk.mesh import Mesh
import sys
import basix
import subprocess
import sys 
from petsc4py import PETSc
from mpi4py import MPI
from flatiron_tk.solver import BlockNonLinearSolver, BlockSplitTree

def no_slip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def zero_pressure(x):
    return np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type)

mesh_file = '../mesh/foc.msh'
mesh = Mesh(mesh_file=mesh_file)

nse = TransientNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()

dt = 0.05
mu = 0.001
rho = 1

u_mag = 4 

nse.set_time_step_size(dt)
nse.set_midpoint_theta(0.5)
nse.set_density(rho)
nse.set_dynamic_viscosity(mu)
nse.set_weak_form(stab=True)

V_u = nse.get_function_space('u').collapse()[0]
V_p = nse.get_function_space('p').collapse()[0]

def inlet_velocity(x):
    # Parabolic profile: u_x = 4 * U_max * y * (H - y) / H^2
    # Assuming inlet along x, y in [0, H], U_max = 10.0, H = 4.1
    values = np.zeros((2, x.shape[1]), dtype=dolfinx.default_scalar_type)
    y = x[1]
    H = 4.1
    U_max = u_mag
    values[0] = 4 * U_max * y * (H - y) / (H ** 2)
    return values

inlet_v = dolfinx.fem.Function(V_u)

# At each time step:
t = 0.0
inlet_v.interpolate(lambda x: inlet_velocity(x))

zero_p = dolfinx.fem.Function(V_p)
zero_p.interpolate(zero_pressure)

zero_v = dolfinx.fem.Function(V_u)
zero_v.interpolate(no_slip)

u_bcs = {
        1: {'type': 'dirichlet', 'value': inlet_v},
        2: {'type': 'dirichlet', 'value': zero_v},
        4: {'type': 'dirichlet', 'value': zero_v},
        5: {'type': 'dirichlet', 'value': zero_v}
        }

p_bcs = {
        3: {'type': 'dirichlet', 'value': zero_p},
        }

bc_dict = {'u': u_bcs, 
           'p': p_bcs}

nse.set_bcs(bc_dict)

nse.set_writer('output-bp', 'bp')

problem = NonLinearProblem(nse)

def set_ksp_u(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setMonitor(ConvergenceMonitor("|----KSPU", verbose=False))
    ksp.setTolerances(max_it=3)
    ksp.pc.setType(PETSc.PC.Type.JACOBI)
    ksp.setUp()

def set_ksp_p(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setMonitor(ConvergenceMonitor("|--------KSPP", verbose=False))
    ksp.setTolerances(max_it=5)
    ksp.pc.setType(PETSc.PC.Type.HYPRE)
    ksp.pc.setHYPREType("boomeramg")
    ksp.setUp()

def set_outer_ksp(ksp):
    ksp.setType(PETSc.KSP.Type.FGMRES)
    ksp.setGMRESRestart(30)
    ksp.setTolerances(rtol=1e-100, atol=1e-10)
    ksp.setMonitor(ConvergenceMonitor("Outer ksp"))

split = {
        'fields': ('u', 'p'),
        'composite_type': 'schur',
        'schur_fact_type': 'full',
        'schur_pre_type': 'a11',
        'ksp0_set_function': set_ksp_u,
        'ksp1_set_function': set_ksp_p
        }

tree = BlockSplitTree(nse, splits=split)
solver = BlockNonLinearSolver(tree, MPI.COMM_WORLD, problem, outer_ksp_set_function=set_outer_ksp)

while t < 1.0:
    print(f'Solving at time t = {t:.2f}')

    # Solve the problem
    solver.solve()

    nse.update_previous_solution()
    nse.write(time_stamp=t)
    
    # Update time
    t += dt 

