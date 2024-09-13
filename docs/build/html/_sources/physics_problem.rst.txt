Physics problem
------------------

PhysicsProblem is a base class for other physics to inherit and define their own physics. This class provides routine functions that are needed by every physics such as function space definition, solution io, etc. This class contains as members, the flatiron_tk.Mesh object the physics is defined on, a ``tag`` which is a name of the physics


=========================
Defining custom physics
=========================
Any physics problem will inherit this class. The developer will need to overload the ``set_weak_form``, ``flux``, and ``get_residue`` methods and properly define the physics of the particular problem. For example, see the `ScalarTransport` physics.


==========================
Defining member functions
==========================
To streamline the physics parameters, we provide a dictionary mapping a string to a function called ``external_function_dict`` as a member function. This dictionary will contain any external functions and variables that will be used to define the physics. For example, if we have a diffusion problem, then the diffusivity of the problem will be encoded in ``external_function_dict``. By doing it this way, we have a centralized location that stores all of the physics definition. Any physics inheriting this class will use the ``set_external_function`` method to assign a function value into this dictionary. 


===================
Class definition
===================
.. autoclass:: flatiron_tk.physics.PhysicsProblem
    :members:
    :undoc-members:


