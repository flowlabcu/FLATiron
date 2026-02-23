import dolfinx
import ufl

from flatiron_tk.mesh import LineMesh
from flatiron_tk.physics import MultiphysicsProblem
from flatiron_tk.physics import PhysicsProblem
from flatiron_tk.solver import NonLinearProblem
from flatiron_tk.solver import BlockSplitTree
from flatiron_tk.solver import BlockNonLinearSolver
from mpi4py import MPI

# Build GP physics
class GalerkinProjection(PhysicsProblem):
    """
    GalerkinProjection field_value = b
    """
    def set_projection_value(self, projection_value):
        self.set_external_function('b', projection_value)

    def flux(self):
        ''''''

    def get_residual(self):
        ''''''

    def set_weak_form(self):
        b = self.external_function('b')
        u = self.get_solution_function()
        w = self.get_test_function()
        self.weak_form = ufl.inner(u-b, w)*self.dx

# Helper function to build GP physics
def build_GP(tag, mesh, val):
    GP = GalerkinProjection(mesh, tag=tag)
    GP.set_element('CG', 1)
    GP.set_projection_value(dolfinx.fem.Constant(mesh.msh, dolfinx.default_scalar_type(val)))
    return GP 

# Create a mesh
mesh = LineMesh(0, 1, 1/5)

# Define three GP physics with different projection values
GP1 = build_GP('A', mesh, val=0)
GP2 = build_GP('B', mesh, val=1)
GP3 = build_GP('C', mesh, val=2)
GPs = [GP1, GP2, GP3]

# Define a multiphysics problem
phs = MultiphysicsProblem(*GPs)
phs.set_element()
phs.build_function_space()
phs.set_weak_form()

# Define the problem 
problem = NonLinearProblem(phs)

# Split the problem into two blocks: (A,C) and (B)
split0 = {'fields': (('A','C'), 'B'), 
            'composite_type': 'schur', 
            'schur_fact_type': 'full', 
            'schur_pre_type': 'a11'}

# Second split: (A,C) into (A) and (C)
split1 = {'fields': ('A','C'), 
            'composite_type': 'schur', 
            'schur_fact_type': 'full', 
            'schur_pre_type': 'a11'}

# Create a list of splits to define a tree structure
splits = [split0, split1]

# Define the tree and solver
tree = BlockSplitTree(phs, splits=splits)
solver = BlockNonLinearSolver(tree, MPI.COMM_WORLD, problem)

# Solve the problem
solver.solve()

# Split the solution into its components 
A, B, C = phs.get_solution_function().split()

# Get the dofs for each component
dofs_A = phs.get_function_space().sub(0).dofmap.list.flatten()
dofs_B = phs.get_function_space().sub(1).dofmap.list.flatten()
dofs_C = phs.get_function_space().sub(2).dofmap.list.flatten()

# Check that the solution is correct
print("Mesh vertices:", mesh.msh.geometry.x.ravel())
print("A values at vertices:", A.x.array[dofs_A])
print("B values at vertices:", B.x.array[dofs_B])
print("C values at vertices:", C.x.array[dofs_C])