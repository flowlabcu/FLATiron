Multi-Physics problem
-------------------------

This is a class that provide an interface for a coupled multi-physics problem. This class provides an interface for physics problem with multiple variables and multiple equation problems.

=========================
Designing multiphysics
=========================
MultiPhysics take in the different PhysicsProblem classes in the constructor, and build a **monolithic** problem based on all of the Physics. The weak formulation that we solve is the sum of all of the weak formulations from the base physics. Here, functions such as the trial, test, and solution functions requires a tag input indicating which variable you are pulling from. These functions will return the **reference** to the specific variable in the monolothic function object. 


===================
Class definition
===================
.. autoclass:: feFlow.physics.MultiPhysicsProblem
    :members:
    :undoc-members:


