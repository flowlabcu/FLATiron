# FLATiron

## 1. About

**FLATiron** is a hierarchical, modular, finite element library for flow physics and transport phenomena. FLATiron - interpreted as FLAT-FE ( **FL**ow and **T**ransport using **F**inite **E**lements) - is named after the famous rock formations in Boulder, Colorado (the site of development of this package). The toolkit is based on the open-source finite element library named `FEniCS`, and leverages the underlying `dolfin` and `Unified Form Language (ufl)` modules heavily in its design. The library has been developed based on research and development activities at the **[FLOWLab](https://www.flowphysicslab.com/)** at *University of Colorado Boulder*. The complete documentation for thei `FLATiron` toolkit is available **[here](https://flatiron-docs.readthedocs.io/en/latest/index.html#)**.

## 2. Features

`FLATiron` is a collection of modular implementations of stabilized finite element methods for fluid flow and advection-diffusion transport processes. The library was designed leveraging the ease of going from mathematical weak formulations of flow and transport problems into high-performance computing modules that is enabled by `FEniCS` and the underlying `ufl` syntaxes. A few key features include:

- Petrov Galerkin stabilization for Navier Stokes and advection-diffusion equations
- Brinkman formulation for immersed porous media
- Jump stabilized formulations for advection-diffusion transport
- Integration of chemical reaction models into transport problems (suited for biochemical cascades)

## 3. Installation instructions

FLATiron is installed in two steps. The library relies on the legacy `FEniCS` library, and the first step is to install legacy FEniCS in your system. This can be done in one of two ways.
- install from source with supplied `FEniCS` source packaged with `FLATiron` toolkit in `extern/src`
- install using legacy build scripts available at <https://fenicsproject.org/download/archive/>
Installation instructions, and additional details are provided in the install page of `FLATiron` documentation available here: **[Install](https://flatiron-docs.readthedocs.io/en/latest/install.html)**

## 4. Demos

For those looking to get started with using `FLATiron`, as well as looking to get started with stabilized finite element simulations for flow and transport problems, we have provided a collection of well documented demos alongwith `FLATiron`. These are curated with complete documentation, which you can access at the following links:
- **Application Demos:**
    - Link: <https://flatiron-docs.readthedocs.io/en/latest/app_demo.html>
    - *These are primarily meant for easy-to-run simulations with minimal knowledge of the underlying finite element mathematics.*
- **Library Demos:**
    - Link: <https://flatiron-docs.readthedocs.io/en/latest/lib_demo.html>
    - *These are primarily meant to help users define their own physics problems using the modules provided.*

## 5. Modules

Module documentation is available in the `FLATiron` documentation page at the following link: **[Modules](https://flatiron-docs.readthedocs.io/en/latest/index.html#modules)**. Comments and feedback on module documentation can be shared by opening an `Issue` on the `GitHub` page, or by emailing the contributors directly below.

## 6. How to cite our code

While `FLATiron` is released as an open source toolkit, there is currently not a single linking article that you can cite to acknowledge use of `FLATiron` in your work. Hence, the best way to cite our tool directly is to link to our Githab Repository.

Below, we have provided a list of research papers that have used the underlying techniques and modules that have gone in to the design and implementation of the `FLATiron` toolkit:

- Teeraratkul, C, Tomaiuolo, M., Stalker, T.J., and Mukherjee, D. *Investigating Clot-flow Interactions By Integrating Intravital ImagingWith In Silico Modeling For Analysis Of Flow, Transport, And Hemodynamic Forces.* **Scientific Reports. 14(1):696. 2024.**

## 7. Questions and issues

For questions and issues, please either directly reach out to contributors below, or report an `Issue` on our GitHub page.

## 8. License

`FLATiron` is released under the **[GNU-LGPL License](https://github.com/flowlabcu/FLATiron/blob/main/LICENSE)**.

## 9. Contributors

`FLATiron` project is created, developed, and maintained by **[FLOWLab](https://www.flowphysicslab.com/)** at the *University of Colorado Boulder* led by Prof. Debanjan Mukherjee (email: debanjan@colorado.edu).

- Dr. Chayut Teeraratkul (email: chayut.teeraratkul@colorado.edu): Lead developer, researcher, and contributor.
- Nick Rovito (email: nick.rovito@colorado.edu): Researcher, and contributor.
