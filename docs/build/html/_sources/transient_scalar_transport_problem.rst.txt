Transient scalar transport
--------------------------------------------------

This class defines the transient version of the scalar transport problem

======================
Strong formulation
======================

The strong form of this now the time-dependent version of the ``ScalarTransport`` problem

.. math::

    \frac{\partial c}{\partial t} + \vec{u}\cdot\nabla c = D\nabla^2c + R \\


==============================================
Initial and Boundary conditions
==============================================

The boundary conditions are the same as the ones defined in the ``ScalarTransport`` class.

Here, the initial condition can be set through the ``set_initial_condition`` method.


=======================
Time discretization
=======================

In this module, the default time discretization is the midpoint :math:`\theta` method defined as follows:

Let subscript :math:`()_0` denote the variable in the previous time step and :math:`()_n` define the variable at the current time step

Find :math:`c_n \in C` and :math:`w \in W` such that

.. math::
    0 = \left(w, \frac{c_n - c_0}{\Delta t} \right) + (1-\theta)\mathcal{L}(c_0; \vec{u}_0, D_0, R_0) + \theta\mathcal{L}(c_n; \vec{u}_n, D_n, R_n) + \sum_{\Omega_e} \int_{\Omega_e} \tau \mathcal{R} \left(\vec{u} \cdot \nabla w \right) d\Omega

where

.. math::
    \mathcal{L}(c; \vec{u}, D, R) := \left(w, \vec{u}\cdot\nabla c\right)_\Omega + \left( \nabla w, D\nabla c \right)_\Omega - \left(w, R\right)_\Omega - \left(w, \vec{h}\cdot \hat{n}\right)_{\Gamma_n} 


where :math:`\mathcal{S}` is the stabilization term for advection dominated problem **CITE**. There are three types of :math:`\tau` that the code provides. These are the `shakib`, `su`, `codina` type. For given a cell diameter :math:`h`, a cell Peclet number :math:`Pe_h=\frac{\lvert \vec{u} \rvert h}{2D}` **CITE Donea**.


============================
Stabilization parameters
============================

Here, we only provide one stabilization constant for the SUPG stabilization based on the ``cordina`` stabilization parameter

.. math::
    \tau_{shakib} = \left( \left( \frac{1}{\theta \Delta t} \right)^2 + \left( \frac{2\lvert \vec{u} \rvert^2}{h}\right)^2 + 9\left(\frac{4D}{h}\right)^2 + R^2 \right)^{-0.5} \\




===================
Class definition
===================
.. autoclass:: feFlow.physics.TransientScalarTransport
    :members:
    :undoc-members:


