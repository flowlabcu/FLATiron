"""
Mark the boundary of a mesh with different tags based on the proximity to cap files.
Note: The walls of the domain are always tagged as 1. Walls DO NOT require an 
stl cap file. The function finds all surface facets of the mesh and 
assigns tags based on the proximity to the midpoints of the 
triangles in the cap files. Any untagged surface facets are 
assigned the default tag of 1 (walls). The function also 
writes the facet markings to a VTU file for inspection.
"""

from flatiron_tk.mesh import mark_xdmf_boundary

MSH_FILE = 'tree_10.msh'
CAPS_DIR = 'tree_10'
OUTPUT_FILE = 'marked_tree_10.xdmf'

mark_xdmf_boundary(MSH_FILE, CAPS_DIR, OUTPUT_FILE)
