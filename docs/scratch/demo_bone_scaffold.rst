Demo bone scaffold
====================

This demo demonstrates how to set up and solve a 3D steady-stokes problem with a porous subdomain. We recommend familiarity with demo_steady_stokes or 
demo_lid_driven_cavity, before starting this demo. 

The domain of this problem is a cylindrical bone sample with channels for fluid placed on top of 
a porous inlet.

		-----------------
		|		| 
		|		|
		|     bone	|
		|		|
		-----------------
		|		|
		| porous media  |
		|		|
		-----------------
		      inlet

An inlet file with that specifies the mesh file, material parameters, and boundary conditions is used. This input file is named bone_scaffold.inp.

.. code-block:: c++

	# Define mesh
	mesh file = scaffold2_4.96um_100kV_otsuFilter_110122/mesh.h5
	output file = scaffold2_4.96um_100kV_otsuFilter_110122/flow_data.h5

	# Physical parameters
	dynamic viscosity = 1.45e-3
	porosity = 0.5
	particle characteristic length = 1

	# Flow conditions
	inlet id = 2
	inlet flow = (0, 0, 0.1)
	outlet id = 3
	outlet pressure = 0
	no slip wall ids = (1, 4)

First, we inport the relevant libraries.

.. code-block:: python

	import numpy as np
	import fenics as fe
	from feFlow.physics import IncompressibleStokes
	from feFlow.mesh import Mesh
	from feFlow.problem import LinearProblem, LinearSolver
	from feFlow.fields import IndicatorFieldScalar
	from feFlow.io import InputObject
	import sys

Next, we read the input file using the ``InputObject`` class. We specify the ``mesh_file`` argument from the ``Mesh`` function to tell feFlow that we are importing an external mesh. 
We also specify the output file. 

.. code-block:: python

	input_file = sys.argv[1]
	input_object = InputObject(input_file)

	mesh_file = input_object('mesh file') 
	mesh = Mesh(mesh_file=mesh_file) 
	output_file = input_object('output file')


Now we can define our fluids and boundary parameters using the input file.

.. code-block:: python

	dynamic_viscosity = input_object('dynamic viscosity')
	porosity = input_object('porosity')
	particle_char_len = input_object('particle characteristic length')

	inlet_id = input_object('inlet id')
	outlet_id = input_object('outlet id')
	inlet_flow = input_object('inlet flow') 
	outlet_pressure = input_object('outlet pressure')
	no_slip_wall_ids = input_object('no slip wall ids')

We need to create a custom class to impliment the porous region. We will use the Brinkman porous media model with permeability based on 
the Kozeny-Karman model for packed spheres. Our custom class, ``BrinkmanSteadyStokes`` will inherit from the ``IncompressibleStokes`` class.

.. code-block:: python

	class BrinkmanSteadyStokes(IncompressibleStokes):

We will define functions to set the porosity and the particle characteristic length.

.. code-block:: python
	
	# within BrinkmanSteadyStokes class
	def set_porosity(self, porosity):
        	self.porosity = porosity
    

	def set_particle_char_len(self, char_len):
        	self.particles_char_len = char_len

Next we define the porous domain. In this problem, the porous domain is spans from 0 <= r <= R of the cylindrical mesh and 
from 1.41457 <= z <= 1.61457. x[2] is the z-direction. 

.. code-block:: python
	
	# within BrinkmanSteadyStokes class
	def in_porous_domain(self, x):
        	return (1.41457 <= x[2] <= 1.61457) 

Now we overload the ``set_weak_form`` defintion from ``IncompressibleStokes``.

.. code-block:: python

	# within BrinkmamSteadyStokes class
	def set_weak_form(self):
        	super().set_weak_form()

We then get the trial and test functions and get the fluid material parameters as defined in the main function.

.. code-block:: python

	# within def set_weak_form
	u, p = fe.TrialFunctions(self.V)
        w, q = fe.TestFunctions(self.V)
        mu = self.dynamic_viscosity

Next, we determine the permeability based on Kozeny-Karman model for packed spheres.

.. code-block:: python
	
	# within def set_weak_form
	phi = self.porosity
        char_len = self.particles_char_len
        K = fe.Constant(char_len**2*phi**3/(150*(1-phi)**2))

Finally we add the brinkman penalty term within the porous domain. We use ``IndicatorFieldScalar`` to determine where to add this term. 
``IndicatorFieldScalar`` is 1 within the porous media and 0 outside of the porous media.

.. code-block:: python

	I = IndicatorFieldScalar(self.in_porous_domain)
        I = fe.interpolate(I, self.V.sub(1).collapse())
        brinkman_term = mu*I/K*fe.inner(w, u)*self.dx
        self.weak_form = self.weak_form + brinkman_term

End of class.

Now we define out physics in the main function. Then we set the finite element
to ``Continuous Galerkin of degree 1``. Finally, we set the function space of the problem.

.. code-block:: python
	
	physics = BrinkmanSteadyStokes(mesh)
	physics.set_element('CG', 1, 'CG', 1)
	physics.set_function_space()

Next, we set the body force term and constants. 

.. code-block:: python

	b = fe.Constant((0, 0, 0)) 
	physics.set_body_force(b)
	physics.set_dynamic_viscosity(dynamic_viscosity)
	physics.set_porosity(porosity)
	physics.set_particle_char_len(particle_char_len)

Now we set the weak form and add stabilization.

.. code-block:: python

	physics.set_weak_form()
	physics.add_stab()

Next, we set the boundary conditions.

.. code-block:: python

	bc_dict = {}
	for wall_id in no_slip_wall_ids:
	    bc_dict[wall_id] = {'type': 'dirichlet', 'field': 'velocity', 'value': 'zero'}
	bc_dict[inlet_id] = {'type': 'dirichlet', 'field': 'velocity', 'value': fe.Constant(inlet_flow)}
	physics.set_bcs(bc_dict)

We explicitly constrain the outlet pressure to be 0.

.. code-block::

	physics.dirichlet_bcs.append(fe.DirichletBC(physics.V.sub(1), fe.Constant(outlet_pressure), mesh.boundary, outlet_id))

Now we set the output writer.

.. code-block:: python

	physics.set_writer(output_file)

We finalize the set-up with setting the type of problem (linear or nonlinear) and linear algebra 
solver. In this case, we 
have a linear PDE as our governing equation, so we set ``LinearProblem`` with the arguement as 
our physics class.

.. code-block:: python

	problem = LinearProblem(physics)
	la_solver = fe.KrylovSolver('bicgstab', 'hypre_amg')
	la_solver.parameters['monitor_convergence'] = True
	solver = LinearSolver(mesh.comm, problem, la_solver)

Once the setup is complete, we can solve and write our solution out.

.. code-block:: python

	solver.solve()
	physics.write()
