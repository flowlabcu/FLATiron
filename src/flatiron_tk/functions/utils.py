import dolfinx 
import flatiron_tk
import numpy as np
import ufl

from mpi4py import MPI

def constant(mesh: 'flatiron_tk.Mesh', value: 'float | tuple') -> dolfinx.fem.Constant:
    """
    Create a dolfinx.fem.Constant with a specified value on a given mesh.

    Args:
        mesh (flatiron_tk.Mesh): The mesh to associate the constant with. 
        value (float or tuple): The value to assign to the constant. Can be a single float or a tuple of floats.

    Returns:
        dolfinx.fem.Constant: The constant object defined on the mesh with the specified value.
    """
    return dolfinx.fem.Constant(mesh.msh, dolfinx.default_scalar_type(value))

def compute_flowrate(flow_physics, id, previous=False):
    """
    Computes the flow rate across a specified boundary in the Navier-Stokes simulation.

    Parameters
    ----------
    flow_physics : TransientNavierStokes, SteadyNavierStokes, or SteadyStokes
        The flow physics object containing the solution and mesh information.
    id : int
        The boundary ID across which to compute the flow rate.  
    previous : bool, optional
        If True, use the previous time step's solution for transient simulations. Default is False.

    Returns
    -------
    flowrate : dolfinx.fem.Function
        A function representing the flow rate across the specified boundary.
    """

    # Get constants from physics object
    u = flow_physics.get_solution_function('u')

    if previous:
        u = flow_physics.previous_solution.sub(0)

    n = flow_physics.mesh.get_facet_normal() 

    form = ufl.inner(u, n) * flow_physics.ds(id)
    flowrate = dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))
    flowrate = MPI.COMM_WORLD.allreduce(flowrate, op=MPI.SUM)
    
    return flowrate
class PointEvaluator:
    """
    Efficiently evaluate finite element functions at user-specified points
    without rebuilding the bounding box tree each time.

    Parameters
    ----------
    mesh : flatiron_tk.Mesh
        The mesh on which the function is defined.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        # Build bounding box tree once
        self.tree = dolfinx.geometry.bb_tree(mesh.msh, mesh.get_tdim())

    def _ensure_3D_point(self, point):
        """
        Convert a 1D, 2D, or 3D point into a 3D NumPy array.
        """
        point = np.array(point, dtype=np.float64).ravel()
        if point.size < 3:
            if MPI.COMM_WORLD.rank == 0:
                print(f'\033[91m[WARNING]\033[0m Point {point} is not 3D. Padding with zeros.')
            point = np.pad(point, (0, 3 - point.size), mode="constant")
        elif point.size > 3:
            raise ValueError("Point must be 1D, 2D, or 3D.")
        return point

    def evaluate_point(self, function, point, show_warning=True):
        """
        Evaluate a Dolfinx Function at a single point in parallel.
        Returns the value or None if the point is outside the mesh.
        
        Parameters
        ----------
        function : dolfinx.fem.Function
            The function to evaluate.
        point : array-like
            Point at which to evaluate. 
        show_warning : bool, optional
            Whether to show a warning if the point is outside the mesh. Default is True.    
        
        Returns
        -------
        value : list or None
            Function value at the point (as a list) or None if outside the mesh.

        """
        point = self._ensure_3D_point(point)

        # Local search
        cells = dolfinx.geometry.compute_collisions_points(self.tree, point)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(
            self.mesh.msh, cells, point[None, :]
        )
        cell_candidates = colliding_cells.links(0)

        local_val = None
        if len(cell_candidates) > 0:
            cell_index = cell_candidates[0]
            local_val = function.eval(point, cell_index)

        # MPI: gather all candidate values
        all_vals = MPI.COMM_WORLD.allgather(local_val)
        # Pick the first non-None value
        for val in all_vals:
            if val is not None:
                return val

        if MPI.COMM_WORLD.rank == 0 and show_warning:
            print(f"\033[91m[WARNING]\033[0m Point {point} is outside the global mesh.")
        return None


    def evaluate_set(self, function, points):
        """
        Evaluate a Dolfinx Function at multiple points in parallel.
        Returns the points (as Nx3 array) and a list of values (or None if outside mesh).

        Parameters
        ----------
        function : dolfinx.fem.Function
            The function to evaluate.
        points : sequence of array-like
            Points at which to evaluate.

        Returns
        -------
        points_3d : np.ndarray
            Nx3 array of points.
        merged : list
            List of function values at each point (None if outside mesh).
        """
        points_3d = np.array([self._ensure_3D_point(p) for p in points], dtype=np.float64)
        n_points = points_3d.shape[0]

        # Local search
        cells = dolfinx.geometry.compute_collisions_points(self.tree, points_3d)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh.msh, cells, points_3d)

        local_results = []
        for i, point in enumerate(points_3d):
            cell_candidates = colliding_cells.links(i)
            if len(cell_candidates) > 0:
                cell_index = cell_candidates[0]
                val = function.eval(point, cell_index)
            else:
                val = None
            local_results.append(val)

        # MPI: gather results from all ranks
        all_results = MPI.COMM_WORLD.allgather(local_results)

        # Merge: pick first non-None value for each point
        merged = []
        for i in range(n_points):
            # all_results is a list of lists, one per rank
            col = [rank_vals[i] for rank_vals in all_results]
            non_none = [v for v in col if v is not None]
            merged.append(non_none[0] if non_none else None)

        return points_3d, merged



