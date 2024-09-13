Creating mesh
---------------------

Mesh files are saved in the ``*.h5`` format. One can generate a mesh from a gmsh :doc:`[4] <references>` script using the ``geo2h5`` script.

``geo2h5`` has 3 input parameters

.. code:: bash

    geo2h5 -d 2/3 -m mesh_file.geo -o output_name

where the ``-d`` flag indicate the dimension of the mesh. The ``-m`` flag supplies the gmsh script, and the ``-o`` flag indicate the output file. This script will output a file call ``output_name.h5`` which can be loaded into the FLATiron workflow.



===================================
Example mesh
===================================

The following is an example mesh file for a simple rectangular mesh. This file is called ``rect.geo`` inside the ``demo/mesh/geo`` directory. Aside from standard gmsh scripting format, it is noted that the mesh version must be saved as version 2.0 as shown in the first line of the meshing script.

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
    Physical Line("1") = {1};
    Physical Line("2") = {2};
    Physical Line("3") = {3};
    Physical Line("4") = {4};
    Line Loop(1) = {1, 2, 3, 4};
    Plane Surface(1) = {1};
    Physical Surface("1") = {1};


To create a mesh, navigate to the directory containing ``rect.geo`` and run

.. code:: bash
    
    geo2h5 -d 2 -m rect.geo -o rect

This will generate ``rect.h5`` as well as other loose files than can be viewed with paraview. If your ``.geo`` script contains ``Physical Line`` and ``Physical Surface``, then ``geo2h5`` will automatically convert them into a marked boundary within the resulting ``.h5`` file. Lastly, to make sure the directory is clean, we provide a simple cleaning script ``clean_mesh_file`` which move each file type into appopriate directories. 

For example, in the current directory structure, ``rect.geo`` is defined inside ``mesh/geo/``, after running ``geo2h5``, you will end up with ``h5``, ``vtu``, ``xml``, etc files. To clean up this directory run

.. code:: bash

    clean_mesh_file ../

This will create directories ``h5/``, ``pvd/``, ``xml/`` at the same level as the ``geo/`` directory, and all of the files will be moved into those directories with the appopriate extension. Note that ``vtu`` and ``pvd`` files are all moved into the ``pvd/`` directory as they are all paraview readable files.

.. code:: bash
    
    cd mesh/geo
    geo2h5 -d 2 -m rect.geo -o rect
    clean_mesh_file ../

and you can locate ``rect.h5`` in ``mesh/h5/rect.h5``. You can also inspect the mesh with paraview from files in ``mesh/pvd/``

