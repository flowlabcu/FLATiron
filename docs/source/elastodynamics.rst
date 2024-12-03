Elastodynamics solver 
==============================================================

This physics class solves the isotropic linear elastic problem. This physics class integrates the resulting governing equation using the generalized alpha scheme. 

The theory of finite element elastodynamics and generalized alpha integration is covered in great detail `here <https://olddocs.fenicsproject.org/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html#erl2002>`_. This documentation will only provide a quick overview on the theory pertaining to the implementation within this library.

It is noted that this class provides a structure for solving classical dynamic structure problems integrated using generalized alpha method. The user may inherit this class and overload the mass, damping, and stress terms for a specific application. 


===========================
Problem definition
===========================
Let :math:`\textbf{u}` be the displacement field and :math:`\textbf{a}:=d^2\textbf{u}/dt^2` be the displacement acceleration within the solid domain. The governing equation is

.. math::

   \rho \textbf{a} = \nabla \cdot \sigma + \rho \textbf{a}_{ext}

where :math:`\rho` is the density and :math:`\sigma` is the stress tensor as a function of :math:`\textbf{u}`. For an isotropic linear elastic problem

.. math::

    \sigma = \lambda tr(\epsilon(\textbf{u}))I + 2\mu\epsilon(\textbf{u}) 

where :math:`I` is the second order identity tensor, :math:`\epsilon(\textbf{u})` is the symmetric strain tensor, and :math:`tr(\textbf{x})` is the trace of :math:`\textbf{x}`

.. math::
   
   \epsilon(\textbf{u}) = \frac{1}{2}\left(\nabla \textbf{u} + \nabla \textbf{u}^T \right)

Parameters :math:`\lambda` and :math:`\mu` are the first and second `Lame's constants <https://en.wikipedia.org/wiki/Lam%C3%A9_parameters>`_. For a suitable function space :math:`V`, define trial and test functions :math:`\textbf{u}\in V` and :math:`\textbf{w}\in V` respectively. The weak formulation of the giverning PDE becomes

.. math::

    0 = \left( \rho \textbf{a}, \textbf{w} \right) + \left( \sigma, \epsilon(\textbf{w}) \right) - \left( \sigma \cdot \hat{n}, \textbf{w} \right)_{\Gamma_n} - \left(\rho \textbf{a}_{ext} ,\textbf{w} \right)_{\Gamma_N}

where :math:`\Gamma_n` is the Neumann boundary. This problem can be written in a commonly used harmonic oscillator form

.. math::

    M(\textbf{a}, \textbf{w}) + C(\textbf{v}, \textbf{w}) + K(\textbf{u},\textbf{w}) - L(\textbf{w}) = 0

where :math:`\textbf{v}:=d\textbf{u}/dt` be the displacement velocity. Within the ``ElastoDynamics`` class, the contributions from :math:`M`, :math:`C`, and :math:`K` can be found in the member functions ``ElastoDynamics.M()``, ``ElastoDynamics.C()`` and ``ElastoDynamics.K()`` respectively.

Comment on damping
-------------------------
The currently presented weak formulation do not have any contribution from the damping term, i.e., :math:`C(\textbf{v}, \textbf{w})=0`. In this physics class, we add a commonly employed `Raleigh damping <https://www.orcina.com/webhelp/OrcaFlex/Content/html/Rayleighdamping.htm>`_ as an example for a damping implementation. By default, the ``ElastoDynamics`` class sets the Raleigh damping parameters to 0 (no damping). The user may `inherit and override <https://www.geeksforgeeks.org/method-overriding-in-python/>`_ the ``ElastoDynamics.C()`` method with an appopriate damping function of your choice as needed.

============================================
Time integration with generalized alpha
============================================
Generalized alpha time integration can be seen as a general form of the mid-point time integration method. Gen-alpha has two parameters :math:`\alpha_m` and :math:`\alpha_f` which represent the midpoint in the acceleration (:math:`\alpha_m`) and position and velocity (:math:`\alpha_f`). Please see `here <https://olddocs.fenicsproject.org/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html#erl2002>`_ for more details. The stability and accuracy of the time integration scheme depends on the choice of :math:`\alpha_m` and :math:`\alpha_f`. Be default, ``ElastoDynamics`` set these constants to :math:`\alpha_m=0.2` and :math:`\alpha_f=0.4` which ensures unconditional stability. The user may change these parameters through the ``ElastoDynamics.set_gen_alpha()`` function.


============================================
I/O
============================================
The ``ElastoDynamics`` class writes the displacement, velocity, and acceleration fields with the physics tag prepended infron of each field file. Optionally, the user can include the projected stress into the I/O process by setting ``write_stress=True`` in the ``ElastoDynamics.set_writer()`` function when initializing the writer.

===================
Class definition
===================
.. autoclass:: flatiron_tk.physics.ElastoDynamics
    :members:
    :undoc-members:


