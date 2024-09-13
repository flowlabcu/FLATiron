Applications
===================

The followings are demo usage for using the application side of FLATiron. This is meant for easy-to-run simulations with minimal knowledge of the underlying finite element mathematics. 

To run the application simulations, simply navigate to apps/ and call the python scripts


.. code:: bash
    
    python3 application_script.py input_file.inp

Currently there are two available application scripts for Incompressible Navier Stokes and Scalar transport problem

The followings are example input files for varying cases of the application simulations

.. toctree::
    :maxdepth: 1
    :caption: Demo input files

    app_steady_adr
    app_transient_adr
    app_stokes_ldc
    app_stokes_pressure_driven
    app_navier_stokes
    app_transient_navier_stokes


