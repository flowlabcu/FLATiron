from flatiron_tk import constant
from flatiron_tk.mesh import CuboidMesh
from flatiron_tk.physics import SteadyScalarTransport
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import NonLinearSolver

mesh = CuboidMesh(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1/20)
stp = SteadyScalarTransport(mesh, tag='c')
stp.set_writer('output', 'pvd')
stp.set_element('CG', 1)
stp.build_function_space()

stp.set_advection_velocity([50.0, 0.0, 0.0])
stp.set_diffusivity(1.0)
stp.set_reaction(0.0)

stp.set_weak_form()
stp.add_stab()  

one_bc = constant(mesh, 1.0)
zero_bc = constant(mesh, 0.0)
bc_dict = {
    1: {'type': 'dirichlet', 'value': one_bc},
    2: {'type': 'dirichlet', 'value': zero_bc},
    3: {'type': 'dirichlet', 'value': zero_bc},
    4: {'type': 'dirichlet', 'value': zero_bc},
    5: {'type': 'dirichlet', 'value': zero_bc},
    6: {'type': 'dirichlet', 'value': zero_bc},
}

stp.set_bcs(bc_dict)

problem = NonLinearProblem(stp)
solver = NonLinearSolver(mesh.msh.comm, problem)

solver.solve()
stp.write()
