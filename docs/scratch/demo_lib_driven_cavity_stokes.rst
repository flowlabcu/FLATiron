Demo lid driven cavity, stokes flow
====================================

This demo shows how to solve the canonical lid driven cavity problem under stokes flow.

**Strong form**

.. math::

	\mu \nabla^2 \underline{u} - \nabla p + \underline{f} = \underline{0} & \\
	\\
	\nabla \cdot \underline{u} = 0 & \\
	\\
	u(x, 0) = u(0, y) = u(L, y) = 0 & \\
	u(x, H) = 1 & \\

First we import the relevant libraries.

.. code-block:: python

	import numpy as np
	import sys
	import fenics as fe
	from feFlow.physics import IncompressibleStokes
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver
	from feFlow.io import InputObject

In this demo, we will import the mesh, boundary conditions, and physical parameters using an input file. This file will contain these contents:

.. code-block:: c++

	# Define mesh
	mesh file = rect.h5
	output file = output/stokes-ldc.pvd

	# What type of element
	# stabilized or taylor-hood
	element type = stabilized

	# Physical parameters
	dynamic viscosity = 1

	# Body force
	body force x = 0.
	body force y = 0.
	body force z = 0.

	# Boundary conditions
	List = BCList 
	boundary_id = 1 ; field = velocity ; value = zero
	boundary_id = 2 ; field = velocity ; value = zero
	boundary_id = 3 ; field = velocity ; value = zero
	boundary_id = 4 ; field = velocity ; value = (1, 0)
	EndList = BCList 

We specify the ``mesh_file`` argument from the ``Mesh`` function to tell feFlow that we are importing an external mesh.

.. code-block:: python

	mesh_file = input_object('mesh file')
	mesh = Mesh(mesh_file=mesh_file)

Next, we define the relevant phyisics. In this case we use the ``IncompressibleStokes`` class. ``IncompressibleStokes`` currently only supports stabilized and taylor-hood 
elements, and requires the user to specify,

.. code-block:: python

	ics = IncompressibleStokes(mesh)
	element_type = input_object('element type')
	if element_type == 'stabilized':
	    ics.set_element('CG', 1, 'CG', 1)
	elif element_type == 'taylor-hood':
	    ics.set_element('CG', 2, 'CG', 1)
	else:
	    raise ValueError('currently only support stabilized or taylor-hood elements for stokes problem')
	ics.set_function_space()

Now, we need to set the coeffifients for each term, and define the body force. These will all be defined in the input file. We convert our 
body force components to a 2D or 3D vector. If the body force component is an expression, we use ``fe.Expression``, else we use ``fe.Constant``.

.. code-block:: python

	mu = input_object('dynamic viscosity')
	bx = input_object('body force x')
	by = input_object('body force y')
	bz = input_object('body force z')
	if ics.mesh.dim == 2:
	    if isinstance(bx, str):
	        b = fe.Expression((bx, by), degree=1)
	    else:
	        b = fe.Constant((bx, by))
	else:
	    if isinstance(bx, str):
	        b = fe.Expression((bx, by, bz), degree=1)
	    else:
	        b = fe.Constant((bx, by, bz))
	ics.set_body_force(b)
	ics.set_dynamic_viscosity(mu) 

Next, we set the weak form and add stabilization.

.. code-block:: python

	ics.set_weak_form()
	ics.add_stab()

We read the boundary conditions from the input file and assign them to a dictionary. The dictionary 'key' is the boundary id, the 'value,' and 
the type and value of the boundary conditions. The types can be either ``dirichlet`` or ``neumann``. **See FEM theory for the difference.**

.. code-block:: python

	bc_inputs = input_object('BCList')
	bc_dict = {}
	for bc in bc_inputs:
	    boundary_id = bc.pop('boundary_id')
	    bc_dict[boundary_id] = bc
	ics.set_bcs(bc_dict)

Now, we setup the output directory. The name of the output file is specified in the input file. 

.. code-block:: python

	output_file = input_object('output file')
	ics.set_writer(output_file)

We finalize the set-up with setting the type of problem (linear or nonlinear) and linear algebra 
solver. In this case, we 
have a linear PDE as our governing equation, so we set ``LinearProblem`` with the arguement as 
our physics class. 

.. code-block:: python

	problem = LinearProblem(ics)
	la_solver = fe.LUSolver()
	solver = LinearSolver(mesh.comm, problem, la_solver)

Finally, we solve the problem and write the solution.

.. code-block:: python

	solver.solve()
	ics.write()


**The full script:**

.. code-block:: python

	import numpy as np
	import sys

	# ------------------------------------------------------- #

	import fenics as fe
	from feFlow.physics import IncompressibleStokes
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver
	from feFlow.io import InputObject


	# Load input object
	input_file = sys.argv[1]
	input_object = InputObject(input_file)

	# Load mesh
	mesh_file = input_object('mesh file')
	mesh = Mesh(mesh_file=mesh_file)

	# Define problem
	ics = IncompressibleStokes(mesh)
	element_type = input_object('element type')
	if element_type == 'stabilized':
	    ics.set_element('CG', 1, 'CG', 1)
	elif element_type == 'taylor-hood':
	    ics.set_element('CG', 2, 'CG', 1)
	else:
	    raise ValueError('currently only support stabilized or taylor-hood elements for stokes problem')
	ics.set_function_space()

	# Set coefficients on each term
	mu = input_object('dynamic viscosity')
	bx = input_object('body force x')
	by = input_object('body force y')
	bz = input_object('body force z')
	if ics.mesh.dim == 2:
	    if isinstance(bx, str):
	        b = fe.Expression((bx, by), degree=1)
	    else:
	        b = fe.Constant((bx, by))
	else:
	    if isinstance(bx, str):
	        b = fe.Expression((bx, by, bz), degree=1)
	    else:
	        b = fe.Constant((bx, by, bz))
	ics.set_body_force(b)
	ics.set_dynamic_viscosity(mu)

	# Set weak form
	ics.set_weak_form()

	# Set stabilization
	ics.add_stab()

	# Set bc
	bc_inputs = input_object('BCList')
	bc_dict = {}
	for bc in bc_inputs:
	    boundary_id = bc.pop('boundary_id')
	    bc_dict[boundary_id] = bc
	ics.set_bcs(bc_dict)

	# Setup io
	output_file = input_object('output file')
	ics.set_writer(output_file)

	# Set problem
	problem = LinearProblem(ics)

	# Set solver
	la_solver = fe.LUSolver()
	solver = LinearSolver(mesh.comm, problem, la_solver)

	# Solve
	solver.solve()
	ics.write()


