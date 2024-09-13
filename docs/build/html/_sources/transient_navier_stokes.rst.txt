Transient Navier Stokes Flow
------------------------------

This class solves the transient Navier Stokes problem. 

=====================
Strong formulation
=====================

Momentum equation

.. math::
    \rho \frac{\partial \vec{u}}{\partial t} + \rho \vec{u} \cdot \nabla \vec{u} = \nabla \cdot \boldsymbol{\sigma} + \vec{b}

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

In this implementation, we the mid-point method for time integration. Let :math:`\mathcal{L}(u, p)` be the weak formulation obtained in the :doc:`steady state Navier Stokes <navier_stokes>`, the weak formulation for the transient problem is

.. math::
    0 = \left( \vec{w}, \frac{\vec{u}^n - \vec{u}^{n-1}}{\Delta t} \right) + \theta\mathcal{L}(\vec{u}^n, p^n) + (1-\theta)\mathcal{L}(\vec{u}^0, p^n)




============================
Stabilization parameters
============================

Stabilization parameter now has time dependence and is defined as

.. math::
    \tau = \left( \left( \frac{1}{\Delta t} \right)^2 + \left(\frac{2|\vec{u}|}{h}\right)^2 + 9\left(\frac{4\nu}{h^2}\right)^2 \right)^{-0.5} \\

In this case, we use the same stabilization parameter for both SUPG and PSPG 


===================
Class definition
===================
.. autoclass:: feFlow.physics.IncompressibleNavierStokes
    :members:
    :undoc-members:



