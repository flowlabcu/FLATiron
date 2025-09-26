import dolfinx 
import flatiron_tk
import numpy as np
import sys

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

def debug_print(*args, **kwargs):
    """
    Debug print function that only prints if the script is run with the
    --debug flag.
    """
    if '--debug' in sys.argv:
        print('\033[92m[DEBUG]\033[0m', *args, **kwargs)

class ParaboloidInletProfile:
    def __init__(self, flow_rate, radius, center, normal):
        """
        Parameters
        ----------
        flow_rate : float
            Volumetric flow rate Q.
        radius : float
            Inlet radius.
        center : array_like
            Center of the inlet circle (3D).
        normal : array_like
            Inlet normal vector (not necessarily unit length).
        """
        self.flow_rate = flow_rate
        self.radius = radius
        self.center = np.array(center, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)  # ensure unit normal

    def update_flow_rate(self, new_flow_rate):
        """Update the flow rate Q for the current timestep."""
        self.flow_rate = new_flow_rate

    @property
    def v_max(self):
        """Compute maximum centerline velocity from flow rate."""
        area = np.pi * self.radius**2
        u_mean = self.flow_rate / area
        return 2 * u_mean

    def __call__(self, x):
        """
        Evaluate velocity profile.

        Parameters
        ----------
        x : np.ndarray of shape (gdim, num_points)

        Returns
        -------
        np.ndarray of shape (gdim, num_points)
            Velocity vectors at the given points.
        """
        dx = x - self.center[:, np.newaxis]
        r = np.linalg.norm(dx, axis=0)

        # Parabolic factor (inside radius)
        factor = np.where(r <= self.radius, 1 - (r / self.radius) ** 2, 0.0)
        velocity_magnitude = self.v_max * factor

        return self.normal[:, np.newaxis] * velocity_magnitude

class ParabolicInletProfile:
    def __init__(self, flow_rate, radius, center, normal):
        """
        Parameters
        ----------
        flow_rate : float
            Volumetric flow rate Q.
        radius : float
            Inlet radius.
        center : array_like
            Center of the inlet (2D).
        normal : array_like
            Inlet normal vector (not necessarily unit length, 2D).
        """
        self.flow_rate = flow_rate
        self.radius = radius
        self.center = np.array(center, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)  # ensure unit normal

    def update_flow_rate(self, new_flow_rate):
        """Update the flow rate Q for the current timestep."""
        self.flow_rate = new_flow_rate

    @property
    def v_max(self):
        """Compute maximum centerline velocity from flow rate."""
        width = 2 * self.radius
        u_mean = self.flow_rate / width
        return 1.5 * u_mean  # For 2D parabolic profile (Poiseuille)

    def __call__(self, x):
        """
        Evaluate velocity profile.

        Parameters
        ----------
        x : np.ndarray of shape (2, num_points)

        Returns
        -------
        np.ndarray of shape (2, num_points)
            Velocity vectors at the given points.
        """
        dx = x - self.center[:, np.newaxis]
        r = np.linalg.norm(dx, axis=0) 
        factor = np.where(r <= self.radius, 1 - (r / self.radius) ** 2, 0.0)
        velocity_magnitude = self.v_max * factor

        return self.normal[:, np.newaxis] * velocity_magnitude