GMSH interface
=======================

FLATiron provides a simple interface between mesh generated from `GMSH <https://gmsh.info/>`_ and FEniCS HDF5 format. The interface is encompassed in the ``geo2h5`` script.

``geo2h5`` has 3 input parameters

.. code:: bash

    geo2h5 -d 2/3 -m mesh_file.geo -o output_name

where the ``-d`` flag indicates the dimension of the mesh (usually 2 or 3). The ``-m`` flag supplies the GMSH ``*.geo`` file, and the ``-o`` flag indicates the name of the output file. This script will output several file format of the same mesh including a ``*.h5`` format primary used in simulations, and a ``*.vtu`` and ``*.pvd`` file format which can be used for initial visualization of the mesh through `paraview <https://www.paraview.org/>`_. 


Example workflow
-----------------

The following is an example GMSH script file for a rectangular mesh. This file is called ``rect.geo`` and can be found inside the ``demo/mesh/geo`` directory. For information regarding the GMSH scripting format, please consult the `GMSH documentation <https://gmsh.info/doc/texinfo/gmsh.html>`_ manual. Aside from standard GMSH scripting format, it is noted that the mesh version **must** be saved as version 2.0 as shown in the first line of the meshing script.

.. code::

    Mesh.MshFileVersion = 2.0;
    // -- Parameters
    w  = 1;
    h  = 0.1;
    dx = h/10;

    // -- Points
    Point(1) = {0, h, 0, dx};
    Point(2) = {w, h, 0, dx};
    Point(3) = {w, 0, 0, dx};
    Point(4) = {0, 0, 0, dx};

    // -- Lines
    Line(4) = {1, 2};
    Line(3) = {2, 3};
    Line(2) = {3, 4};
    Line(1) = {4, 1};

    // -- Add physical ids for the lines and surface
    Physical Line(1) = {1};
    Physical Line(2) = {2};
    Physical Line(3) = {3};
    Physical Line(4) = {4};
    Line Loop(1) = {1, 2, 3, 4};
    Plane Surface(1) = {1};
    Physical Surface("1") = {1};


To create a mesh, navigate to the directory containing ``rect.geo`` and run the following command

.. code:: bash
    
    geo2h5 -d 2 -m rect.geo -o rect

This command will generate ``rect.h5`` as well as other loose files than can be viewed with paraview. If your ``.geo`` script contains ``Physical Line`` (2D mesh) or ``Physical Surface`` (3D mesh), then ``geo2h5`` will automatically convert them into a marked boundary within the resulting ``.h5`` file. The marking ID will follow the integer value provided in the ``Physical Line/Surface (#)`` line within the GMSH script. The marking values can be inspected by using paraview to read the ``*_boundaries.pvd`` file. Lastly, to make sure the directory structure is clean, we provide a simple housekeeping script called ``clean_mesh_file`` which move each file type into appopriate directories. 

For example, in the current directory structure, ``rect.geo`` is defined inside ``mesh/geo/``, after running ``geo2h5``, you will end up with ``h5``, ``vtu``, ``xml``, etc files. To clean up this directory run from within the ``mesh/geo/`` directory

.. code:: bash

    clean_mesh_file ../

This will create directories ``h5/``, ``pvd/``, ``xml/`` at the same level as the ``geo/`` directory, and all of the files will be moved into those directories with the appopriate extension. Note that ``vtu`` and ``pvd`` files are all moved into the ``pvd/`` directory as they paired together.


In summary the full workflow will be as follows

.. code:: bash
    
    cd demo/mesh/geo
    geo2h5 -d 2 -m rect.geo -o rect
    clean_mesh_file ../

now, you can locate use ``rect.h5`` in ``demo/mesh/h5/rect.h5`` for your simulation, and visualize the mesh and the boundary markings in ``demo/mesh/pvd/``.

