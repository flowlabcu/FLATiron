Navier Stokes Flow
-------------------------

This class solves the steady Navier Stokes problem. In this case, the solver uses the Cauchy stress formulation of the problem

=====================
Strong formulation
=====================

Define the Cauchy stress for incompressible flow as

.. math::
    \boldsymbol{\sigma} := -p\textbf{I} + 2\mu(\nabla \vec{u} + \nabla \vec{u}^T)

Momentum equation

.. math::
    \rho \vec{u} \cdot \nabla \vec{u} = \nabla \cdot \boldsymbol{\sigma} + \vec{b}

Continuity

.. math::
    \nabla \cdot \vec{u} = 0

where :math:`\vec{u}` and `p` are the velocity and pressure field respectively with the constants :math:`\rho` and :math:`\mu` are the fluid density and dynamic viscosity respectively.


=======================
Boundary conditions
=======================

Fixed value boundary condition

.. math::
    \vec{u} = \vec{u}_D \;\forall \vec{x} \in \Gamma_D

Traction boundary condition

.. math::
    \boldsymbol{\sigma} \cdot \hat{\textbf{n}} = \vec{t} \;\;\forall \vec{x} \in \Gamma_N



=======================
Weak formulation
=======================

The weak formulation is stated as follows:

For the same test/trial function spaces as :doc:`stokes_flow`


.. math::
    0 = \left(\vec{w}, \rho \vec{u} \cdot \nabla \vec{u} \right)_\Omega + \left(\nabla\vec{w}, \boldsymbol{\sigma}\right)_\Omega - \left(q, \nabla \cdot \vec{u}\right)_\Omega + (\vec{w}, \vec{t})_{\Gamma_N} - (\vec{w}, \vec{b}_\Omega) + S_{SUPG} + S_{PSPG}

Where the stabilization terms are defined as

.. math::
    S_{SUPG} = \sum_{\Omega_e} \int_{\Omega_e} \tau \vec{u} \cdot \nabla \vec{w} \cdot \vec{R} d\Omega  

.. math::
    S_{PSPG} = \sum_{\Omega_e} \int_{\Omega_e} \frac{1}{\rho} \tau \nabla q \cdot \vec{R} d\Omega  

where :math:`\vec{R}` is the residue of the strong form



============================
Stabilization parameters
============================

For linear element, the stabilization is defined according to :doc:`[1] <references>` as

.. math::
    \tau = \left( \left(\frac{2|\vec{u}|}{h}\right)^2 + 9\left(\frac{4\nu}{h^2}\right)^2 \right)^{-0.5} \\

In this case, we use the same stabilization parameter for both SUPG and PSPG 


===================
Class definition
===================
.. autoclass:: flatiron_tk.physics.SteadyIncompressibleNavierStokes
    :members:
    :undoc-members:



