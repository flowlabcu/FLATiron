Stokes flow
-------------------------

This class solves the Stokes flow problem which is the linearized form of the Navier-Stokes equation

=====================
Strong formulation
=====================

Momentum equation

.. math::
    \nabla p = \nu \nabla^2 \vec{u} + \vec{b}

Continuity

.. math::
    \nabla \cdot \vec{u} = 0

where :math:`\vec{u}` and `p` are the velocity and pressure field respectively with the constants :math:`\nu` is the kinematic viscosity. Note that pressure here is the scaled pressure. True pressure :math:`p_{true} = p/\rho` where :math:`\rho` is the density. :math:`\vec{b}` is the external body force


=======================
Boundary conditions
=======================


Fixed value boundary condition

.. math::
    \vec{u} = \vec{u}_D \;\forall \vec{x} \in \Gamma_D

(pseudo) Traction boundary condition

.. math::
    -p\textbf{n} + \nu(\textbf{n} \cdot \nabla)\vec{u} = \vec{t} \;\;\forall \vec{x} \in \Gamma_N



=======================
Weak formulation
=======================

The weak formulation is stated as follows:

For the velocity trial function

.. math::
    \mathcal{U} := \{\vec{U} \in H^1(\Omega) | \vec{u} = \vec{u}_D \;\text{on}\; \Gamma_D\}

and the corresponding test function

.. math::
    \mathcal{W} := \{\vec{w} \in H^1(\Omega) | \vec{w} = 0 \;\text{on}\; \Gamma_D\}

And the pressure space

.. math::
    \mathcal{Q} := \mathcal{L}_2(\Omega)

for external body force :math:`\vec{b}` and surface traction :math:`\vec{t}`, find :math:`\vec{u} \in \mathcal{U}` and :math:`p \in \mathcal{Q}` with the corresponding test functions :math:`\vec{w} \in \mathcal{W}` and :math:`q \in \mathcal{Q}` such that

.. math::
    0 = \left(\nabla\vec{w}, \nu\nabla\vec{u}\right)_\Omega - \left( \nabla \cdot \vec{w}, p \right)_\Omega - \left(\vec{w}, \vec{b}\right)_\Omega - \left(q, \nabla \cdot \vec{u}\right)_\Omega + (\vec{w}, \vec{t})_{\Gamma_N} + \sum_{\Omega_e} \int_{\Omega_e}  \tau \nabla q \cdot \nabla p d\Omega



============================
Stabilization parameters
============================

For linear element, the stabilization is defined according to :doc:`[1] <references>` as

.. math::
    \tau = \frac{1}{3} \frac{h^2}{4\nu} \\

===================
Class definition
===================
.. autoclass:: flatiron_tk.physics.StokesFlow
    :members:
    :undoc-members:



