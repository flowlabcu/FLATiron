import dolfinx
import numpy as np
import ufl

from flatiron_tk.mesh import RectMesh
from flatiron_tk.functions import build_field_scalar_function
from flatiron_tk.functions import PointEvaluator
from mpi4py import MPI

def test_scalar_function():
    # -------------------------
    # Domain and indicator function
    # -------------------------
    domain = RectMesh(0.0, 0.0, 10.0, 10.0, 1/40)
    fictitious = RectMesh(1.0, 1.0, 6.0, 6.0, 1/40)
    inside_value = 1.0
    outside_value = 0.0
    I = build_field_scalar_function(domain, fictitious, inside_value, outside_value)

    evaluator = PointEvaluator(domain)

    # -------------------------
    # Evaluate scalar function with evaluate_point
    # -------------------------
    points = [np.array([9.0, 9.0, 0.0]), np.array([2.0, 3.0, 0.0])]
    for pt in points:
        val = evaluator.evaluate_point(I, pt)
        val_scalar = val[0] if isinstance(val, (list, np.ndarray)) else val
        # Expected: 1.0 inside fictitious, 0.0 outside
        expected = 1.0 if 1.0 <= pt[0] <= 6.0 and 1.0 <= pt[1] <= 6.0 else 0.0
        assert np.allclose(val_scalar, expected), f"Unexpected scalar value at {pt}: got {val_scalar}, expected {expected}"

    # -------------------------
    # Evaluate scalar function with evaluate_set
    # -------------------------
    points_set = [np.array([9.0, 9.0, 0.0]), np.array([2.0, 3.0, 0.0])]
    pts_out, vals_out = evaluator.evaluate_set(I, points_set)
    for pt, val in zip(pts_out, vals_out):
        val_scalar = val[0] if isinstance(val, (list, np.ndarray)) else val
        expected = 1.0 if 1.0 <= pt[0] <= 6.0 and 1.0 <= pt[1] <= 6.0 else 0.0
        assert np.allclose(val_scalar, expected), f"Unexpected scalar value at {pt}: got {val_scalar}, expected {expected}"

def test_vector_function():
    # -------------------------
    # Vector function test
    # -------------------------
    mesh = RectMesh(0.0, 0.0, 1.0, 1.0, 0.1)
    V = dolfinx.fem.functionspace(mesh.msh, ("CG", 1, (mesh.get_gdim(),)))
    u = dolfinx.fem.Function(V)
    x = ufl.SpatialCoordinate(mesh.msh)
    expr = dolfinx.fem.Expression(2 * x, V.element.interpolation_points())
    u.interpolate(expr)

    test_points = [np.array([0.2, 0.3]), np.array([0.9, 0.9]), np.array([1.5, 1.5])]
    peval = PointEvaluator(mesh)

    for p in test_points:
        val = peval.evaluate_point(u, p)
        if val is None:
            continue
        val = np.array(val)
        expected = np.pad(2 * p, (0, mesh.get_gdim() - len(p)))
        assert np.allclose(val, expected), f"Unexpected vector value at {p}: got {val}, expected {expected}"

    # Vector function test
    peval = PointEvaluator(mesh)
    vector_pts_out, vector_vals_out = peval.evaluate_set(u, test_points)

    for pt, val in zip(vector_pts_out, vector_vals_out):
        if val is None:
            continue

        val = np.array(val)
        pad_len = mesh.get_gdim() - len(pt)
        if pad_len > 0:
            expected = np.pad(2 * pt[:len(pt)], (0, pad_len))
        else:
            expected = 2 * pt[:mesh.get_gdim()]  # truncate if point is longer than gdim

        assert np.allclose(val, expected), f"Unexpected vector value at {pt}: got {val}, expected {expected}"
