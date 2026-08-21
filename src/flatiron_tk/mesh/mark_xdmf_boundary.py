import os
import copy

import numpy as np
import dolfinx
import meshio

from dolfinx.mesh import locate_entities_boundary
from scipy.spatial import KDTree
from .mesh import Mesh

def _num_facets(msh):
    return msh.topology.index_map(msh.topology.dim - 1).size_local

def _write_facet_markings(msh, boundary):
    if msh.topology.dim == 3:
        facet_type = 'triangle'
    elif msh.topology.dim == 2:
        facet_type = 'line'
    else:
        raise ValueError('Only works on 2D and 3D simplex mesh')

    mesh_x = msh.geometry.x
    facet_conn = msh.topology.connectivity(msh.topology.dim - 1, 0)
    num_facets_per_cell = facet_conn.offsets[1] - facet_conn.offsets[0]

    facet_array = copy.deepcopy(facet_conn.array)
    facet_array.resize((_num_facets(msh), num_facets_per_cell))

    markings = np.zeros(_num_facets(msh), dtype=np.int32)
    markings[boundary.indices] = boundary.values

    facet_mesh = meshio.Mesh(mesh_x, [(facet_type, facet_array)], cell_data={'facet_ids': [markings]})
    facet_mesh.write('facet_markings.vtu')
    print('Facet markings is written to facet_markings.vtu')

def mark_xdmf_boundary(msh_file, caps_dir, output_file='marked_mesh.xdmf'):
    '''
    Marks the boundary facets of a mesh and assigns tags based on proximity to cap midpoints,
    then writes the marked mesh to an XDMF file.

    Parameters
    -----------
        msh_file : str, Path to the *.msh mesh file (without caps marked).
        caps_dir : str, Path to the directory containing cap *.stl files.
        output_file : str, Path to write the marked mesh. Default is 'marked_mesh.xdmf'.

    Returns
    -----------
        None: The function writes the marked mesh (and facet markings) to file.

    The function performs the following steps:
    1. Reads the mesh with the flatiron_tk Mesh class.
    2. Marks all facets as boundary facets.
    3. Reads cap files and computes the midpoints of the triangles in each cap.
    4. Constructs KD-trees for the midpoints of the triangles in each cap.
    5. Iterates over each boundary facet and assigns a tag based on the proximity to the cap midpoints.
    6. Writes the marked mesh to an XDMF file.
    '''

    # Read the mesh using the flatiron_tk Mesh class
    mesh_obj = Mesh(mesh_file=msh_file)
    msh = mesh_obj.msh
    tdim = mesh_obj.get_tdim()
    fdim = mesh_obj.get_fdim()

    msh.topology.create_connectivity(fdim, tdim)

    # Mark all the boundary facets
    def all_boundary(x):
        return np.full(x.shape[1], True, dtype=bool)
    boundary_facets = locate_entities_boundary(msh, fdim, all_boundary)

    # Create an array that will hold the boundary tags
    # We default to 1, which is the tag for the walls (if a wall 'cap' is not provided)
    boundary_tags = np.ones_like(boundary_facets, dtype=np.int32)

    # Compute the minimum element size
    entity_indices = np.zeros(msh.topology.index_map(fdim).size_local, dtype=np.int32)
    h = dolfinx.cpp.mesh.h(msh._cpp_object, tdim, entity_indices)
    hmin = min(h)
    eps = hmin / 10

    # Read the cap files and compute the midpoints of the triangles in each cap (stl only)
    cap_files = [f for f in os.listdir(caps_dir) if os.path.isfile(os.path.join(caps_dir, f)) and f.endswith('.stl')]
    if mesh_obj.comm.rank == 0:
        print(f'Found {len(cap_files)} cap files in {caps_dir}')
        print(f'Cap files: {cap_files}')

    caps = [meshio.read(os.path.join(caps_dir, cap_file)) for cap_file in cap_files]
    tris = [cap.points[cap.cells[0].data] for cap in caps]
    midpoints = [np.mean(tri, axis=1) for tri in tris]

    # Construct KD-trees for the midpoints of the triangles in each cap
    trees = [KDTree(midpoint) for midpoint in midpoints]

    # Iterate over each boundary facet and assign a tag based on the proximity to the cap midpoints
    for count, f in enumerate(boundary_facets):
        facet_midpoint = np.mean(msh.geometry.x[msh.topology.connectivity(fdim, 0).links(f)], axis=0)

        # Query the KD-trees to find the cap that the facet is closest to
        # eps = (hmin/10) is the maximum distance from the facet midpoint to a cap triangle midpoint
        for tree_index, tree in enumerate(trees):
            # The cap index is tree_index + 2 because the wall is tagged as 1 (if we do not provide a wall)
            cap_index = tree_index + 2
            ind = tree.query_ball_point(facet_midpoint, eps)
            if len(ind) > 0:
                boundary_tags[count] = cap_index
                break

    if mesh_obj.comm.rank == 0:
        for tree_idx in range(len(trees)):
            print(f'Boundary ID: {tree_idx + 2}, Boundary File: {cap_files[tree_idx]}')

    # Mark boundary facets on the flatiron_tk Mesh object
    mesh_obj.boundary = dolfinx.mesh.meshtags(msh, fdim, boundary_facets, boundary_tags)
    mesh_obj.boundary.name = 'facet_tags'

    # Mark all cells (subdomain) with tag 0
    cell_indices = np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)
    cell_tags = np.zeros_like(cell_indices, dtype=np.int32)
    mesh_obj.subdomain = dolfinx.mesh.meshtags(msh, tdim, cell_indices, cell_tags)
    mesh_obj.subdomain.name = 'cell_tags'

    # Write the facet markings to a VTU file for inspection
    _write_facet_markings(msh, mesh_obj.boundary)

    # Save the marked mesh as XDMF
    with dolfinx.io.XDMFFile(mesh_obj.comm, output_file, 'w') as xdmf_file:
        xdmf_file.write_mesh(msh)
        xdmf_file.write_meshtags(mesh_obj.boundary, msh.geometry)
        xdmf_file.write_meshtags(mesh_obj.subdomain, msh.geometry)

    if mesh_obj.comm.rank == 0:
        print(f'Marked mesh is written to {output_file}')
