Transient scalar transport
----------------------------------

This is a demo of how to solve a transient advection-diffusion-reaction problem using an input file for the prebuild input-defined solvers. These examples are intended to be used as a "plug and play" option for new users. Explanations of the inputs are described below. 

The input file is largely identical to the :doc:`app_steady_adr` input file. This doc will hilight the additional inputs that are required for the transient version of the simulation. The example input file can be found in ``app/inputs/transient_adr.inp`` 

===========================
Running the demo
===========================

To run the transient transport problem, you can use the same ``scalar_transport.py`` code that was used in :doc:`app_steady_adr` with the new input file

.. code:: bash

    python3 scalar_transport.py inputs/transient_adr.inp

===========================
Input Descriptions
===========================

Setup for importing a mesh, i/o, flow physics type, flow phyisics properties, stabilization, boundary conditions, and solver type share the same inputs as in the steady-ADR setup. Please reference :doc:`app_steady_adr` for details. 

We only need to change the output prefix to ``adr_transient``, and the flow physics type to ``transient adr``.

.. code-block::

	# Output directory prefix
	output prefix = adr_transient
	output type = pvd

	# Set flow physics type
	transport physics type = transient adr

When solving a transient problem, we need to define the time-dependent parameters. ``time step size`` is describes the discrete time interval for solving tranient problems.  Here we set a time step of 0.01. This input should be a constant scalar quantity. ``time span`` is the total simulation time. Here we simulate an ADR problem on the time interval [0,1]. This input should be a constant scalar quantity. ``save every`` descirbes which time steps will be saved for visualization. Here we set the problem so each time step is saved. This input should be a constant scalar quantity.

.. code-block:: 

	# Time dependent variables
	time step size = 1e-2
	time span = 1
	save every = 1

We set the initial condition using the ``initial condition`` input. ``initial condition`` has three inputs and should take the form: (ic_file, ic_group, ic_time_id). ``ic_file`` is a ``*.h5`` file type describing the initial value of ``ic_group``. ``ic_group`` is the variable of the weak form we are applying the initial condition to. 

In the case where the initial condition file is from a transient simulation, the user can provide ``ic_time_id`` which indicate the timestep the user would like to use as the initial condition. 

If no initial condition is provided, the initial condition is assumed to be zero everywhere.



