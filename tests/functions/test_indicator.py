import pytest
import numpy as np
import dolfinx
from flatiron_tk.functions import build_field_scalar_function
from flatiron_tk.mesh import RectMesh

def test_indicator(float_equal):
    """
    Testing the indicator function in a fictisious domain scheme.

    Author: JHolmes
    """

    # Define the domain and fictitious region
    domain = RectMesh(0.0, 0.0, 10.0, 10.0, 1/40)
    fictitious = RectMesh(1.0, 1.0, 6.0, 6.0, 1/40)

    # Build a scalar function that is 1 inside the fictitious region and 0 outside
    inside_value = 1.0
    outside_value = 0.0
    I = build_field_scalar_function(domain, fictitious, inside_value, outside_value)

    # Defining points to evaluate within the boarder domain 
    points = [np.array([9.0, 9.0, 0.0], dtype=np.float64), np.array([2.0, 3.0, 0.0], dtype=np.float64)]
    output_values = []

    # Evaluating the indicator function at each point
    for pt in points:
        tree = dolfinx.geometry.bb_tree(domain.msh, domain.get_tdim())
        cells = dolfinx.geometry.compute_collisions_points(tree, pt)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(domain.msh, cells, np.array([pt], dtype=np.float64))
        cell_candidates = colliding_cells.links(0)
        if len(cell_candidates) > 0:
                cell_index = cell_candidates[0]
                output_values.append(I.eval(pt, cell_index)[0])

    assert float_equal(output_values[0], 0.0)
    assert float_equal(output_values[1], 1.0)
