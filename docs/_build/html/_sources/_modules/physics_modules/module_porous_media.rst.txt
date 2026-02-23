^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Brinkman Navier-Stokes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transient Brinkman Navier-Stokes
----------------------------------
.. automodule:: flatiron_tk.physics.transient_brinkman_navier_stokes
   :members:
   :undoc-members:
   :show-inheritance:

This class solves the transient Brinkman Navier Stokes problem. The Brinkman equations are a modification of the Navier-Stokes equations to 
account for flow in porous media by adding a Darcy drag term to the momentum equation.

=====================
Strong formulation
=====================

Momentum equation

.. math::
    \rho \frac{\partial \textbf{u}}{\partial t} + \rho \textbf{u} \cdot \nabla \textbf{u} = \nabla \cdot \boldsymbol{\sigma} + \textbf{b} + \frac{\mu}{K} \textbf{u}

Continuity

.. math::
    \nabla \cdot \textbf{u} = 0

where :math:`\textbf{u}` and `p` are the velocity and pressure field respectively with the constants :math:`\rho` and :math:`\mu` are the fluid density and dynamic viscosity respectively.

=======================
Boundary conditions
=======================

Fixed value boundary condition

.. math::
    \textbf{u} = \textbf{u}_D \;\forall \textbf{x} \in \Gamma_D

Traction boundary condition

.. math::
    \boldsymbol{\sigma} \cdot \hat{\textbf{n}} = \textbf{t} \;\;\forall \textbf{x} \in \Gamma_N

=======================
Weak formulation
=======================

In this implementation, we the mid-point method for time integration. Let :math:`\mathcal{L}(u, p)` be the weak 
formulation obtained in the :doc:`module_navier-stokes`, the weak formulation for the transient problem is

.. math::
    0 = \left( \textbf{w}, \frac{\textbf{u}^n - \textbf{u}^{n-1}}{\Delta t} \right) + \theta \left( \mathcal{L}(\textbf{u}^n, p^n) + \textbf{i} \frac{\mu}{\kappa}(\textbf{u}^n, \textbf{w}) \right) + (1-\theta)\left( \mathcal{L}(\textbf{u}^0, p^n) + \textbf{i} \frac{\mu}{\kappa}(\textbf{u}^n, \textbf{w}) \right)

============================
Stabilization parameters
============================

Stabilization parameter now has time dependence and is defined as

.. math::
    \tau = \left( \left( \frac{1}{\Delta t} \right)^2 + \left(\frac{2|\textbf{u}|}{h}\right)^2 + 9\left(\frac{4\nu}{h^2}\right)^2 \right)^{-0.5} \\

In this case, we use the same stabilization parameter for both SUPG and PSPG 