Scalar transport
-------------------------

This class defines the scalar transport problem, the so-called Advection-Diffusion-Reaction problem

========================
Strong formulation
========================

Define a total computational domian as :math:`\Omega` and the Dirichlet and Neumann boundaries :math:`\Gamma_D` and :math:`\Gamma_N` where :math:`\Gamma_D \cup \Gamma_D = \partial\Omega` and :math:`\Gamma_D \cap \Gamma_D = 0`. Let the trial function space be defined s

.. math::

    \vec{u}\cdot\nabla c = D\nabla^2c + R \\

where :math:`c` is the concentration field, :math:`\vec{u}` is the velocity, :math:`D` is diffusivity, and :math:`R` is the reaction.

=======================
Boundary conditions
=======================

Fixed value boundary condition

.. math::
    c = c_D \; \forall \vec{x} \in \Gamma_D

Diffusive flux boundary condition

.. math::
    \nabla c \cdot \hat{n} = \vec{h} \cdot \hat{n} \; \forall \vec{x} \in \Gamma_N

where :math:`\hat{n}` is the unit normal to :math:`\Gamma_N`


=======================
Weak formulation
=======================

The weak formulation is stated as follows:

.. math::
    C := \{c \in H^1(\Omega) | c = c_D \;\text{on}\; \Gamma_D\}

and the corresponding test function

.. math::
    W := \{w \in H^1(\Omega) | w = 0 \;\text{on}\; \Gamma_D\}

The weak formulation, for :math:`c \in C` and :math:`w \in W`

.. math::
    0 = \left(w, \vec{u}\cdot\nabla c\right)_\Omega + \left( \nabla w, D\nabla c \right)_\Omega - \left(w, R\right)_\Omega - \left(w, \vec{h}\cdot \hat{n}\right)_{\Gamma_n} + \sum_{\Omega_e} \int_{\Omega_e} \tau \mathcal{R} \left(\vec{u} \cdot \nabla w \right) d\Omega

where ther residue :math:`\mathcal{R}` is

.. math::
    \mathcal{R} = \vec{u} \cdot \nabla c - D\nabla^2c - R 

where :math:`\mathcal{S}` is the stabilization term for advection dominated problem :doc:`[1] <references>`. There are three types of :math:`\tau` that the code provides. These are the `shakib`, `su`, `codina` type. For given a cell diameter :math:`h`, a cell Peclet number :math:`Pe_h=\frac{\lvert \vec{u} \rvert h}{2D}`  


============================
Stabilization parameters
============================

Stabilization parameters :math:`\tau` are predefined values adapted from :doc:`[1] <references>` via the ``get_stab_constant(tau_type)`` method. The ``tau_type`` parameter can be either ``shakib``, ``su``, or ``codina``, and are defined as

.. math::
    \tau_{shakib} = \left( \left( \frac{2\lvert \vec{u} \rvert^2}{h}\right)^2 + 9*\left(\frac{4D}{h}\right)^2 + R^2 \right)^{-0.5} \\

.. math::
    \tau_{su} =  \frac{h}{2 \lvert \vec{u} \rvert} \left( coth(Pe_h) - \frac{1}{Pe_h}\right) \\

.. math::
    \tau_{codina} =  \frac{h}{2 \lvert \vec{u} \rvert} \left(1 + \frac{1}{Pe_h} + \frac{hR}{2\lvert \vec{u} \rvert}\right)^{-1} \\




===================
Class definition
===================
.. autoclass:: feFlow.physics.ScalarTransport
    :members:
    :undoc-members:


