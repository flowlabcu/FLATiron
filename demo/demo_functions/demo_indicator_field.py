import dolfinx

from flatiron_tk.functions import build_field_scalar_function
from flatiron_tk.functions import build_rank_indicator_function
from flatiron_tk.mesh import RectMesh
from mpi4py import MPI

# Define the domain and fictitious region
domain = RectMesh(0.0, 0.0, 10.0, 10.0, 1/20)
fictitious = RectMesh(2.0, 2.0, 8.0, 8.0, 1/20)

# Build a scalar function that is 1 inside the fictitious region and 0 outside
inside_value = 1.0
outside_value = 0.0
scalar_function = build_field_scalar_function(domain, fictitious, inside_value, outside_value)

# Write the scalar function to an XDMF file
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'scalar_function.xdmf', 'w') as xdmf:
    xdmf.write_mesh(domain.msh)
    xdmf.write_function(scalar_function)

# This function builds a rank indicator function for the mesh 
rank_indicator_function = build_rank_indicator_function(domain)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'rank_indicator_function.xdmf', 'w') as xdmf:
    xdmf.write_mesh(domain.msh)
    xdmf.write_function(rank_indicator_function)

