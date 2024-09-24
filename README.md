# FLATiron

## 1. About

**FLATiron** is a hierarchical, modular, finite element library for flow physics and transport phenomena. FLATiron - interpreted as FLAT-FE ( **FL**ow and **T**ransport using **F**inite **E**lements) - is named after the famous rock formations in Boulder, Colorado (the site of development of this package). The toolkit is based on the open-source finite element library named `FEniCS`, and leverages the underlying `dolfin` and `Unified Form Language (ufl)` modules heavily in its design. The library has been developed based on research and development activities at the *FLOWLab* at *University of Colorado Boulder* (https://www.flowphysicslab.com/).

## 2. Features

FLATiron is a collection of modular implementations of stabilized finite element methods for fluid flow and advection-diffusion transport processes. The library was designed leveraging the ease of going from mathematical weak formulations of flow and transport problems into high-performance computing modules that is enabled by `FEniCS` and the underlying `ufl` syntaxes. A few key features include:

- Petrov Galerkin stabilization for Navier Stokes and advection-diffusion equations
- Brinkman formulation for immersed porous media
- Jump stabilized formulations for advection-diffusion transport

## 3. Installation instructions

FLATiron is installed in two steps. The library relies on legacy FEniCS library, and the first step is to install legacy FEniCS in your system.
- extern/src
- legacy build scripts available at https://fenicsproject.org/download/archive/

## 4. Demos

A collection of well documented demos are curated with complete documentation at:

## 5. Module

Module documentation is available at the following location:

Comments and feedback on module documentation can be shared by opening an `Issue` on the `GitHub` page, or by emailing the contributors directly below.

## 6. How to cite our code


## 7. Questions and issues


## 8. License

GNU LGPL

## 9. Contributors

Chayut and Nick
