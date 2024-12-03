.. FLATiron documentation master file, created by
   sphinx-quickstart on Thu Feb 29 15:42:38 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FLATiron's documentation!
========================================

Welcome to FLATiron's documentation!

FLow And Transport Finite element or FLATiron (Fe == iron) is a toolkit used to solve coupled flow physics problems via the Finite Element Method. The finite element backend of FLATiron is based on legacy FEniCS. FLATiron is a hierarchical, modular, finite element library for flow physics and transport phenomena. This toolkit is based on the open-source finite element library named `FEniCS <https://fenicsproject.org/>`_, and leverages the underlying dolfin and Unified Form Language (ufl) modules heavily in its design. The library has been developed based on research and development activities at the FLOWLab at University of Colorado Boulder.

.. _introduction:

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   install
   meshing

..
    ########################################

.. _demos:

.. toctree::
    :maxdepth: 2
    :caption: Demos

    app_demo
    lib_demo

..
    ########################################


.. _modules:

.. toctree::
    :maxdepth: 2
    :caption: API references

    physics
    mesh
    io


