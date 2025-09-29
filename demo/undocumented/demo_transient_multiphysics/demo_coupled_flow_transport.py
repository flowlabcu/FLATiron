import numpy as np

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()
MPI = import_mpi4py()

from flatiron_tk.physics import TransientNavierStokes
from flatiron_tk.physics import TransientMultiPhysicsProblem
from flatiron_tk.physics import TransientScalarTransport
from flatiron_tk.solver  import ConvergenceMonitor
from flatiron_tk.solver import NonLinearSolver
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import NonLinearProblem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mesh_file = '../mesh/foc.msh'
mesh = Mesh(mesh_file=mesh_file)

# Physical parameters
viscosity = 0.001
density = 1.0
u_bar = 6.0 # Inlet velocity magnitude

# Domain dimensions (from original mesh generation context)
diameter = 0.2
box_height = 4.1 * diameter
box_length = 22.0 * diameter

# Time discretization parameters
dt = 0.005
t_start = 0.0
t_end = 1.0
t_theta = 0.5 # Theta parameter for theta-Galerkin method

reynolds = density * u_bar * diameter / viscosity
if rank == 0:
    print(f"The Problem Reynolds Number is: {reynolds:.2f}")

id_inlet = 1
id_bottom = 2
id_outlet = 3
id_top = 4
id_cylinder = 5

nse = TransientNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.set_time_step_size(dt)
nse.set_midpoint_theta(0.5)
nse.set_density(density)
nse.set_dynamic_viscosity(viscosity)

adr = TransientScalarTransport(mesh)
adr.set_tag('c')
adr.set_element('CG', 1)
adr.set_time_step_size(dt)
adr.set_diffusivity(viscosity, viscosity)
adr.set_reaction(0.0, 0.0)

coupled_physics = TransientMultiPhysicsProblem(nse, adr)
coupled_physics.set_element()
coupled_physics.build_function_space()

u = nse.get_solution_function('u')
u0 = ufl.split(coupled_physics.sub_physics[0].previous_solution)[0]

adr.set_advection_velocity(u0, u0)
nse_options = {'stab': True}
adr_options = {'stab': True}
coupled_physics.set_weak_form(nse_options, adr_options)

V_u = nse.get_function_space('u').collapse()[0]
V_p = nse.get_function_space('p').collapse()[0]
V_c = adr.get_function_space().collapse()[0]

# Define boundary conditions functions
def no_slip_bc(x):
    """No-slip boundary condition for velocity (u=0)."""
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def pressure_bc(x):
    """Pressure boundary condition (p=0)."""
    return np.zeros(x.shape[1])

def inlet_bc(x):
    """Parabolic inlet velocity profile."""
    vals = np.zeros((mesh.msh.geometry.dim, x.shape[1]))
    vals[0] = 4.0 * u_bar * x[1] * (box_height - x[1]) / (box_height * box_height)
    vals[1] = 0.0
    return vals

inlet_v = dolfinx.fem.Function(V_u)
inlet_v.interpolate(lambda x: inlet_bc(x))

zero_p = dolfinx.fem.Function(V_p)
zero_p.interpolate(pressure_bc)

zero_v = dolfinx.fem.Function(V_u)
zero_v.interpolate(no_slip_bc)

c_bc = dolfinx.fem.Function(V_c)
c_bc.interpolate(lambda x: np.ones(x.shape[1]))

u_bcs = {
        id_inlet: {'type': 'dirichlet', 'value': inlet_v},

        id_top: {'type': 'dirichlet', 'value': zero_v},
        id_bottom: {'type': 'dirichlet', 'value': zero_v},
        id_cylinder: {'type': 'dirichlet', 'value': zero_v},
        }

p_bcs = {
        id_outlet: {'type': 'dirichlet', 'value': zero_p},
        }

c_bcs = {
        id_inlet: {'type': 'dirichlet', 'value': c_bc},
}

bc_dict = {'u': u_bcs, 
           'p': p_bcs,
           'c': c_bcs}

coupled_physics.set_bcs(bc_dict)

problem = NonLinearProblem(coupled_physics)

def my_custom_ksp_setup(ksp):
    # Get the PETSc.Options object
    opts = PETSc.Options()

    # Get the option prefix for this specific KSP object
    # This ensures that these options only apply to 'ksp'
    option_prefix = ksp.getOptionsPrefix()

    # Set KSP type to GMRES
    opts[f"{option_prefix}ksp_type"] = "gmres"

    # Set PC type to LU
    opts[f"{option_prefix}pc_type"] = "lu"

    # Set the direct solver for LU factorization to Mumps
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

    # Set tolerances and max iterations for the LINEAR solver (KSP)
    opts[f"{option_prefix}ksp_rtol"] = 1e-7
    opts[f"{option_prefix}ksp_atol"] = 1e-10
    opts[f"{option_prefix}ksp_max_it"] = 500 # Max iterations for the linear solve
    ksp.setFromOptions()
    ksp.setMonitor(ConvergenceMonitor('ksp'))

# Instantiate NonLinearSolver, passing all NewtonSolver parameters explicitly
solver = NonLinearSolver(
    mesh.msh.comm, 
    problem, 
    outer_ksp_set_function=my_custom_ksp_setup,
    rtol=1e-12, # NewtonSolver (nonlinear) relative tolerance
    atol=1e-10, # NewtonSolver (nonlinear) absolute tolerance
    max_it=100, # NewtonSolver (nonlinear) max iterations
    convergence_criterion="incremental", # NewtonSolver criterion
    report=True # Enable reporting for NewtonSolver
)

coupled_physics.set_writer('output','pvd')
t = 0.0
while t < 10.0:
    print(f'Solving at time t = {t:.2f}')

    # Solve the problem
    solver.solve()

    coupled_physics.update_previous_solution()
    coupled_physics.write(time_stamp=t)
    
    # Update time
    t += dt
