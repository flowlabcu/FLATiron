========================================
Demo: Marking XDMF Boundaries
========================================

This demo illustrates how to use ``mark_xdmf_boundary`` to tag the boundary of a mesh
that was generated without pre-tagged boundary surfaces (e.g. a raw ``*.msh`` file with
an untagged outer surface).

Given a mesh file and a directory of cap ``*.stl`` files (one per inlet/outlet), the
function locates every boundary facet, finds its distance to each cap's triangle
midpoints, and assigns a tag: any facet close to a cap is tagged with that cap's ID
(starting at 2), and every other facet defaults to tag 1 (the wall). The marked mesh
is written to XDMF, and a VTU of the facet tags is written alongside it for visual
inspection in Paraview.

.. code-block:: python

    from flatiron_tk.mesh import mark_xdmf_boundary

    MSH_FILE = 'tree_10.msh'
    CAPS_DIR = 'tree_10'
    OUTPUT_FILE = 'marked_tree_10.xdmf'

    mark_xdmf_boundary(MSH_FILE, CAPS_DIR, OUTPUT_FILE)

``CAPS_DIR`` should contain one ``*.stl`` file per cap surface (e.g. inlets/outlets);
each cap is assigned a tag in the order it is discovered, starting at 2. Walls do not
need a cap file — any boundary facet not matched to a cap is tagged 1 automatically.
