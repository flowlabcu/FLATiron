Demo Steady-ADR Input-Defined
=============================

This is a demo of how to solve a steady ADR problem using an input file for the prebuilt input-defined solvers. These examples are intended to be used as a "plug and play" option for new users. Explanations of the inputs are described below.

The following is the example input for a simple steady state advection-diffusion-reaction problem. The contents of the input file steady_adr.inp are given as:

.. code-block:: bash

	# Mesh file
	mesh file = ../mesh/h5/rect.h5

	# Output directory prefix
	output prefix = adr_steady
	output type = pvd

	# Set flow physics type
	transport physics type = steady adr

	# Physical properties
	diffusivity = 1
	flow velocity = (1, 0)
	reaction = 0.0

	# SUPG stabilization
	add supg = true

	# Boundary conditions
	gc_inlet = (1, 0)
	zero = 0.0
	BC1 = (1, gradient value, gc_inlet)
	BC2 = (3, fixed value, zero)

	# Linear solver
	solver type = direct


=============================
Input descriptions
=============================

The mesh file is the path to the ``h5`` format file containg the mesh

.. code-block::

	mesh file = /path_mesh_file/mesh.h5

The directory containing the output files is defined in the ``output prefix`` variable. Additionally, the user must provide an output file type, either ``h5`` or ``pvd`` files. ``pvd`` will output a series of VTK files for each time step and MPI partition. These are readily readable from paraview. The ``h5`` format will contain all of the output data within a single file. These ``h5`` output format are ideal for large parallel jobs. To convert the ``h5`` format to paraview readable files, see the ``h5_to_pvd`` function in :doc:`h5_mod <h5_mod>`

In this example, we will name our output "adr_steady" as a VTK file (.pvd).

.. code-block::

	output prefix = adr_steady
	output type = pvd

We now define the flow physics type. For steady simulations use "steady adr" and "transient adr" for unsteady.

.. code-block::

	transport physics type = steady adr

Next, we define the physical properties of the simulation. For ADR problems, supply a diffusivity, velocity, and reaction. In steady ADR, constant scalar values diffusivity and reaction variables are supported. The advection velocity is a list input. The number of entries must match the problem dimension. In this problem, we describ a advection-diffusion problem with no reaction

.. code-block::

	diffusivity = 1
	flow velocity = (1, 0)
	reaction = 0.0

Next, we can add stabilization. Currently, only streamline upwind Petrov-Galerkin stabilization (SUPG) is supported. Set the stabilization variable to true or false

.. code-block::

	add supg = true

Now we must define boundary conditions. All boundary condition variables must start with "BC" to be recognized by the interpreter. **Note that each active BC must have a unique name otherwise feFlow will overwrite the duplicating BCs**.

The format of the BC goes as follows: ``(Boundary ID, BC_type, value variable)``

    ``Boundary ID`` refer to the id of the boundary this BC is being applied to.

    ``BC_type`` refers to the type of BC. It can be either ``gradient value`` or ``fixed value`` corresponding to the Neuman and Dirichlet boundary condition respectively. For more information, please see :doc:`here <scalar_transport_problem>`

    ``value variable`` refers to the name of the variable defined within the input file. In this case, we defined ``gc_inlet = (1, 0)``, and BC1 has ``gc_inlet`` as its variable name. The same pattern is repeated for BC2 and variable name ``zero``

.. code-block::

	gc_inlet = (1, 0)
	zero = 0.0
	BC1 = (1, gradient value, gc_inlet)
	BC2 = (3, fixed value, zero)

Finally we set the linear algebra solver to the direct solver. 

.. code-block::

	solver type = direct

