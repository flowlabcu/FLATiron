^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Scalar-Transport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Steady Scalar-Transport
-------------------------
.. automodule:: flatiron_tk.physics.steady_scalar_transport
   :members:
   :undoc-members:
   :show-inheritance:

This class defines the scalar transport problem, the so-called Advection-Diffusion-Reaction problem

========================
Strong formulation
========================

Define a total computational domian as :math:`\Omega` and the Dirichlet and Neumann 
boundaries :math:`\Gamma_D` and :math:`\Gamma_N` where :math:`\Gamma_D \cup \Gamma_D = \partial\Omega` and :math:`\Gamma_D \cap \Gamma_D = 0`. Let the trial 
function space be defined s

.. math::

    \textbf{u}\cdot\nabla c = D\nabla^2c + R \\

where :math:`c` is the concentration field, :math:`\textbf{u}` is the velocity, :math:`D` is diffusivity, and :math:`R` is the reaction.

=======================
Boundary conditions
=======================

Fixed value boundary condition

.. math::
    c = c_D \; \forall \textbf{x} \in \Gamma_D

Diffusive flux boundary condition

.. math::
    \nabla c \cdot \hat{n} = \textbf{h} \cdot \hat{n} \; \forall \textbf{x} \in \Gamma_N

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
    0 = \left(w, \textbf{u}\cdot\nabla c\right)_\Omega + \left( \nabla w, D\nabla c \right)_\Omega - \left(w, R\right)_\Omega - \left(w, \textbf{h}\cdot \hat{n}\right)_{\Gamma_n} + \sum_{\Omega_e} \int_{\Omega_e} \tau \mathcal{R} \left(\textbf{u} \cdot \nabla w \right) d\Omega

where ther residue :math:`\mathcal{R}` is

.. math::
    \mathcal{R} = \textbf{u} \cdot \nabla c - D\nabla^2c - R 

where :math:`\mathcal{S}` is the stabilization term for advection dominated problem. 
There are three types of :math:`\tau` that the code provides. These are the `shakib`, `su`, `codina` type. 
For given a cell diameter :math:`h`, a cell Peclet number :math:`Pe_h=\frac{\lvert \textbf{u} \rvert h}{2D}`  


============================
Stabilization parameters
============================

Stabilization parameters :math:`\tau` are predefined values  via the ``get_stab_constant(tau_type)`` method. The ``tau_type`` 
parameter can be either ``shakib``, ``su``, or ``codina``, and are defined as

.. math::
    \tau_{shakib} = \left( \left( \frac{2\lvert \textbf{u} \rvert^2}{h}\right)^2 + 9*\left(\frac{4D}{h}\right)^2 + R^2 \right)^{-0.5} \\

.. math::
    \tau_{su} =  \frac{h}{2 \lvert \textbf{u} \rvert} \left( coth(Pe_h) - \frac{1}{Pe_h}\right) \\

.. math::
    \tau_{codina} =  \frac{h}{2 \lvert \textbf{u} \rvert} \left(1 + \frac{1}{Pe_h} + \frac{hR}{2\lvert \textbf{u} \rvert}\right)^{-1} \\


Transient Scalar-Transport
-----------------------------
.. automodule:: flatiron_tk.physics.transient_scalar_transport
   :members:
   :undoc-members:
   :show-inheritance:

This class defines the transient version of the scalar transport problem

======================
Strong formulation
======================

The strong form of this now the time-dependent version of the ``TransientScalarTransport`` problem

.. math::

    \frac{\partial c}{\partial t} + \textbf{u}\cdot\nabla c = D\nabla^2c + R \\


==============================================
Initial and Boundary conditions
==============================================

The boundary conditions are the same as the ones defined in the ``TransientScalarTransport`` class.

Here, the initial condition can be set through the ``set_initial_condition`` method.


=======================
Time discretization
=======================

In this module, the default time discretization is the midpoint :math:`\theta` method defined as follows:

Let subscript :math:`()_0` denote the variable in the previous time step and :math:`()_n` define the variable at the current time step

Find :math:`c_n \in C` and :math:`w \in W` such that

.. math::
    0 = \left(w, \frac{c_n - c_0}{\Delta t} \right) + (1-\theta)\mathcal{L}(c_0; \textbf{u}_0, D_0, R_0) + \theta\mathcal{L}(c_n; \textbf{u}_n, D_n, R_n) + \sum_{\Omega_e} \int_{\Omega_e} \tau \mathcal{R} \left(\textbf{u} \cdot \nabla w \right) d\Omega

where

.. math::
    \mathcal{L}(c; \textbf{u}, D, R) := \left(w, \textbf{u}\cdot\nabla c\right)_\Omega + \left( \nabla w, D\nabla c \right)_\Omega - \left(w, R\right)_\Omega - \left(w, \textbf{h}\cdot \hat{n}\right)_{\Gamma_n} 


where :math:`\mathcal{S}` is the stabilization term for advection dominated problem. 
There are three types of :math:`\tau` that the code provides. These are the `shakib`, `su`, `codina` type. 
For given a cell diameter :math:`h`, a cell Peclet number :math:`Pe_h=\frac{\lvert \textbf{u} \rvert h}{2D}`.


============================
Stabilization parameters
============================

Here, we only provide one stabilization constant for the SUPG stabilization based on the ``cordina`` stabilization parameter

.. math::
    \tau_{shakib} = \left( \left( \frac{1}{\theta \Delta t} \right)^2 + \left( \frac{2\lvert \textbf{u} \rvert^2}{h}\right)^2 + 9\left(\frac{4D}{h}\right)^2 + R^2 \right)^{-0.5} \\


===================
Class definition
===================
.. autoclass:: flatiron_tk.physics.TransientScalarTransport
    :members:
    :undoc-members:

