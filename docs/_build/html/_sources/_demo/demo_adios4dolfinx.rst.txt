=========================================
Demo: Using ADIOS4DOLFINx
=========================================
This demo shows how to use the ADIOS2 file format with DOLFINx [dokken2024]_.

ADIOS4DOLFINx allows for solution functions to be read from and written to ADIOS2 binary-pack files. This is particularly useful 
for checkpointing long simulations or for file transfer in high-performance computing environments.

Converting to Visualization Formats
-------------------------------------
We include two utilities to convert ADIOS2 binary-pack files to XDMF and PVD formats for visualization in Paraview.
The following code snippet demonstrates how to convert a binary-pack file to XDMF and PVD formats. 

First, we run a script that generates the ADIOS2 binary-pack file using MPI with 2 processes. 

.. code-block:: python

    import flatiron_tk
    import subprocess

    subprocess.run(['mpirun', '-n', '2', 'python3', '_run_nse.py'])

Then, we use the `flatiron_tk.bp_to_xdmf` and `flatiron_tk.bp_to_pvd` functions to perform the conversions. We must 
specify the input and output file names, the name of the function to be converted, the time index to include, 
and the element family, degree, and shape of the function. The `time_id` parameter can be set to 'all' to include all time steps, 
or to a specific integer index.

.. code-block:: python

    flatiron_tk.bp_to_xdmf('output-bp/u.bp', 'output-xdmf/u.xdmf', name='u', time_id='all',
                        element_family='CG', element_degree=1, element_shape='vector')

    flatiron_tk.bp_to_pvd('output-bp/u.bp', 'output-pvd/u.pvd', name='u', time_id='all',
                        element_family='CG', element_degree=1, element_shape='vector')


Reading a Function from ADIOS2 Binary-Pack File
------------------------------------------------------------------------
The following code snippet demonstrates how to read a solution function from an ADIOS2 
binary-pack and how to manipulate it in your simulations.

First, we run a script that generates the ADIOS2 binary-pack file using MPI with 2 processes. 

.. code-block:: python

    subprocess.run(['mpirun', '-n', '2', 'python3', '_run_nse.py'])

Next, we read the function from the binary-pack file using the `flatiron_tk.read_function_from_bp` function. Here, we 
specify the input file name, the time index to read (using -1 to read the last time step), the name of the function,
and the element family, degree, and shape of the function.

.. code-block:: python

    u = flatiron_tk.bp_read_function('output-bp/u.bp', time_id=-1, name='u', 
                                element_family='CG', element_degree=1, 
                                element_shape='vector')

We can then manipulate the function `u` as needed in our simulation. In this example, we scale the solution by a factor of -0.1. 

.. code-block:: python

    u_neg = dolfinx.fem.Function(u.function_space)
    u_neg.name = 'u_neg'
    u_neg.x.array[:] = -0.1 * u.x.array[:]
    u_neg.x.scatter_forward()

    with dolfinx.io.VTKFile(MPI.COMM_WORLD, 'output-read/u.pvd', 'w') as vtk:
        vtk.write_function(u, 0.0)
        
    with dolfinx.io.VTKFile(MPI.COMM_WORLD, 'output-read/u_neg.pvd', 'w') as vtk:
        vtk.write_function(u_neg, 0.0)

Full Scripts
----------------

**Converting ADIOS2 Binary-Pack to XDMF and PVD**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import flatiron_tk
    import subprocess

    subprocess.run(['mpirun', '-n', '2', 'python3', '_run_nse.py'])

    # Convert ADIOS2 BP file to XDMF and PVD formats for visualization
    flatiron_tk.bp_to_xdmf('output-bp/u.bp', 'output-xdmf/u.xdmf', name='u', time_id='all',
                        element_family='CG', element_degree=1, element_shape='vector')

    flatiron_tk.bp_to_pvd('output-bp/u.bp', 'output-pvd/u.pvd', name='u', time_id='all',
                        element_family='CG', element_degree=1, element_shape='vector')


**Reading and Manipulating a Function from ADIOS2 Binary-Pack File**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import dolfinx
    import flatiron_tk
    import subprocess
    from mpi4py import MPI

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


References
----------
.. [dokken2024] Dokken, J. S. (2024). ADIOS4DOLFINx: A framework for checkpointing in FEniCS. Journal of Open Source Software, 9(96), 6451. https://doi.org/10.21105/joss.06451
