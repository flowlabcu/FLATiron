import flatiron_tk
import subprocess

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()
MPI = import_mpi4py()

# Run a simple Navier-Stokes example to generate data
subprocess.run(['mpirun', '-n', '2', 'python3', '_run_nse.py'])

# Read in the velocity field from the ADIOS2 output file
u = flatiron_tk.bp_read_function('output-bp/u.bp', time_id=-1, name='u', 
                                element_family='CG', element_degree=1, 
                                element_shape='vector')

# Create a new function that is a manipulation of the read-in function
u_neg = dolfinx.fem.Function(u.function_space)
u_neg.name = 'u_neg'
u_neg.x.array[:] = -0.1 * u.x.array[:]
u_neg.x.scatter_forward()

# Save both the read-in function and the manipulated function to VTK files
with dolfinx.io.VTKFile(MPI.COMM_WORLD, 'output-read/u.pvd', 'w') as vtk:
    vtk.write_function(u, 0.0)
    
with dolfinx.io.VTKFile(MPI.COMM_WORLD, 'output-read/u_neg.pvd', 'w') as vtk:
    vtk.write_function(u_neg, 0.0)

# View Glyphs in Paraview to see manipulations post read
print('View Glyphs in Paraview to see manipulations post read')
