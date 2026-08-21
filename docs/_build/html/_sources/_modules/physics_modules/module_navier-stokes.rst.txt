Steady Navier-Stokes
-------------------------
.. automodule:: flatiron_tk.physics.steady_navier_stokes
   :members:
   :undoc-members:
   :show-inheritance:

This class solves the steady Navier Stokes problem. In this case, the solver uses the Cauchy stress formulation of the problem

=====================
Strong formulation
=====================

Define the Cauchy stress for incompressible flow as

.. math::
    \boldsymbol{\sigma} := -p\textbf{I} + 2\mu(\nabla \textbf{u} + \nabla \textbf{u}^T)

Momentum equation

.. math::
    \rho \textbf{u} \cdot \nabla \textbf{u} = \nabla \cdot \boldsymbol{\sigma} + \textbf{b}

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

The weak formulation is stated as follows:

For the same test/trial function spaces as :doc:`module_stokes`, the weak formulation for the Navier-Stokes problem is


.. math::
    0 = \left(\textbf{w}, \rho \textbf{u} \cdot \nabla \textbf{u} \right)_\Omega + \left(\nabla\textbf{w}, \boldsymbol{\sigma}\right)_\Omega - \left(q, \nabla \cdot \textbf{u}\right)_\Omega + (\textbf{w}, \textbf{t})_{\Gamma_N} - (\textbf{w}, \textbf{b}_\Omega) + S_{SUPG} + S_{PSPG}

Where the stabilization terms are defined as

.. math::
    S_{SUPG} = \sum_{\Omega_e} \int_{\Omega_e} \tau \textbf{u} \cdot \nabla \textbf{w} \cdot \textbf{R} d\Omega  

.. math::
    S_{PSPG} = \sum_{\Omega_e} \int_{\Omega_e} \frac{1}{\rho} \tau \nabla q \cdot \textbf{R} d\Omega  

where :math:`\textbf{R}` is the residue of the strong form



============================
Stabilization parameters
============================

For linear element, the stabilization is defined as

.. math::
    \tau = \left( \left(\frac{2|\textbf{u}|}{h}\right)^2 + 9\left(\frac{4\nu}{h^2}\right)^2 \right)^{-0.5} \\

In this case, we use the same stabilization parameter for both SUPG and PSPG 


Transient Navier-Stokes
-----------------------------
.. automodule:: flatiron_tk.physics.transient_navier_stokes
   :members:
   :undoc-members:
   :show-inheritance:

This class solves the transient Navier Stokes problem. 

=====================
Strong formulation
=====================

Momentum equation

.. math::
    \rho \frac{\partial \textbf{u}}{\partial t} + \rho \textbf{u} \cdot \nabla \textbf{u} = \nabla \cdot \boldsymbol{\sigma} + \textbf{b}

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

In this implementation, we the mid-point method for time integration. Let :math:`\mathcal{L}(u, p)` be the weak formulation obtained in the above steady forumation, the 
weak formulation for the transient problem is

.. math::
    0 = \left( \textbf{w}, \frac{\textbf{u}^n - \textbf{u}^{n-1}}{\Delta t} \right) + \theta\mathcal{L}(\textbf{u}^n, p^n) + (1-\theta)\mathcal{L}(\textbf{u}^0, p^n)

============================
Stabilization parameters
============================

Stabilization parameter now has time dependence and is defined as

.. math::
    \tau = \left( \left( \frac{1}{\Delta t} \right)^2 + \left(\frac{2|\textbf{u}|}{h}\right)^2 + 9\left(\frac{4\nu}{h^2}\right)^2 \right)^{-0.5} \\

In this case, we use the same stabilization parameter for both SUPG and PSPG 

