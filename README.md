# FLATiron

## 1. About

**FLATiron** is a hierarchical, modular, finite element library for flow physics and transport phenomena. FLATiron - interpreted as FLAT-FE ( **FL**ow and **T**ransport using **F**inite **E**lements) - is named after the famous rock formations in Boulder, Colorado (the site of development of this package). The toolkit is based on the open-source finite element library named `FEniCS`, and leverages the underlying `dolfinx` and `Unified Form Language (ufl)` modules heavily in its design. The library has been developed based on research and development activities at **[FLOWLab](https://www.flowphysicslab.com/)** at the *University of Colorado Boulder*. The complete documentation for the `FLATiron` toolkit is available **[here](https://flatiron-docs.readthedocs.io/en/latest/index.html#)**.

## 2. Features

`FLATiron` is a library of modular implementations of stabilized finite element methods for fluid flow and transport processes. The library was designed by leveraging the ease of going from mathematical weak formulations of flow and transport problems to high-performance computing modules that is enabled by the finite element library `FEniCSx`; the underlying `dolfinx`, `ffcx`, and `basix` libraries; and the associated `ufl` syntaxes. A few key features include:

- Petrov-Galerkin stabilization for Navier-Stokes and advection-diffusion equations.
- Brinkman formulation for immersed porous media.
- Jump stabilized formulations for advection-diffusion transport.
- Integration of chemical reaction models in transport problems (suited for biochemical cascades).
- Fluid-thermal coupling and bioheat transport modeling capabilities.

## 3. Installation instructions

`FLATiron` is installed in two steps.

- The library relies on the `dolfinx v0.9.0` library. It's recommended that `dolfinx` is installed using a package manager like `Anaconda`. This is the first requisite step.
- Thereafter, the `gmsh` for meshing needs to be installed. An additional library for accessing advanced file I/O capabilities, named `adios4dolfinx` is needed.
- Finally, the `FLATiron` module can be installed using `pip`.

Detailed installation instructions, and additional details for `dolfinx` base libraries are provided on the install page of `FLATiron` documentation available here: **[Install](https://flatiron-docs.readthedocs.io/en/latest/installation.html)**

## 4. Demos

For those looking to get started with using `FLATiron`, as well as looking to get started with stabilized finite element simulations for flow and transport problems, we have provided a collection of well documented demos along with `FLATiron`. These are curated with complete documentation, which you can access at the following links:

- **Library Demos:**
    - Link: <https://flatiron-docs.readthedocs.io/en/latest/lib_demo.html>
    - *These are primarily meant to help users define their own physics problems using the modules provided.*

## 5. Modules

Module documentation is available at the `FLATiron` documentation page at the following link: **[Modules](https://flatiron-docs.readthedocs.io/en/latest/index.html#modules)**. Comments and feedback on module documentation can be shared by opening an `Issue` on the `GitHub` page, or by emailing the contributors directly below.

## 6. How to cite our code

While `FLATiron` is released as an open source toolkit, there is currently not a single linking article that you can cite to acknowledge use of `FLATiron` in your work. Hence, the best way to cite our tool is to link directly to our Github repository.

Below, we have provided a list of research papers that have described the underlying techniques and modules which have informed the design and implementation of the `FLATiron` toolkit:

- Venkatesh, S., Teeraratkul, C., Rovito, N., Mukherjee, D., and Lynch, M. E., 2025, “High-Fidelity Computational Fluid Dynamics Modeling to Simulate Perfusion through a Bone-Mimicking Scaffold,” **Computers in Biology and Medicine**, 186, p. 109637.
‌
- Teeraratkul, C., Tomaiuolo, M., Stalker, T.J., and Mukherjee, D. “Investigating Clot-flow Interactions By Integrating Intravital ImagingWith In Silico Modeling For Analysis Of Flow, Transport, And Hemodynamic Forces.” **Scientific Reports. 14(1):696. 2024.**

- Rovito, N., and Mukherjee, D. In Silico Analysis Of Flow-mediated Drug Transport For Thrombolytic
Therapy In Acute Ischemic Stroke. **In: ASME International Mechanical Engineering Congress and Exposition, Vol. 88667, pp. V008T10A026. 2025**.
‌
## 7. Questions and issues

For questions and issues, please either directly reach out to the contributors below, or report an `Issue` on our GitHub page.

## 8. License

`FLATiron` is released under the **[GNU-LGPL License](https://github.com/flowlabcu/FLATiron/blob/main/LICENSE)**.

## 9. Contributors

`FLATiron` project is created, developed, and maintained by the **[FLOWLab](https://www.flowphysicslab.com/)** at the *University of Colorado Boulder* led by Prof. Debanjan Mukherjee (email: debanjan@colorado.edu).

- Dr. Chayut Teeraratkul (email: chayut.teeraratkul@colorado.edu): Lead developer, researcher, and contributor.
- Nick Rovito (email: nick.rovito@colorado.edu): Lead developer, researcher, and contributor.
- Jessica Holmes (email: jessica.holmes-1@colorado.edu): Researcher, and contributer.
