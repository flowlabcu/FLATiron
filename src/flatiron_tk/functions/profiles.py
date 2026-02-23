import numpy as np

class ParaboloidInletProfile:
    def __init__(self, flow_rate, radius, center, normal, mesh=None):
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
        center = np.array(center, dtype=float)
        if center.shape[0] == 2:
            center = np.pad(center, (0, 1))

        self.center = center
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

import numpy as np

class ParabolicInletProfile:
    def __init__(self, flow_rate, radius, center, normal, mesh):
        """
        Parameters
        ----------
        flow_rate : float
            Volumetric flow rate Q.
        radius : float
            Inlet "radius" (half height for 2D channel).


        center : array_like of length 3
            Center of the inlet (3D).
        normal : array_like of length 3
            Inward normal vector (flow direction, 3D).
        """
        self.flow_rate = flow_rate
        self.radius = radius
        self.mesh = mesh
        
        center = np.array(center, dtype=float)
        normal = np.array(normal, dtype=float)
        if center.shape[0] == 2: center = np.pad(center, (0, 1))
        if normal.shape[0] == 2: normal = np.array(normal, dtype=float)
        
        self.center = center
        self.normal = normal / np.linalg.norm(normal)  # unit normal vector
        
        # Compute tangent vector (perpendicular to normal) for variation direction
        # To find tangent, pick an arbitrary vector not parallel to normal:
        arbitrary = np.array([0.0, 0.0, 1.0])
        tangent = np.cross(self.normal, arbitrary)
        self.tangent = tangent / np.linalg.norm(tangent)  # unit tangent vector

    def update_flow_rate(self, new_flow_rate):
        """Update the flow rate Q for the current timestep."""
        self.flow_rate = new_flow_rate

    @property
    def v_max(self):
        """Compute maximum centerline velocity from flow rate.

        Assuming a 2D parabolic profile: u_mean = Q / width,
        and u_max = 1.5 * u_mean for Poiseuille flow.
        """
        width = 2 * self.radius
        u_mean = self.flow_rate / width
        return 1.5 * u_mean

    def __call__(self, x):
        """
        Evaluate velocity profile.

        Parameters
        ----------
        x : np.ndarray of shape (3, num_points)
            Points where velocity is evaluated.

        Returns
        -------
        np.ndarray of shape (3, num_points)
            Velocity vectors at the given points.
        """
        if x.shape[0] != 3:
            raise ValueError("Input points x must be 3D arrays with shape (3, num_points)")
        
        # Compute vector from center to points
        dx = x - self.center[:, np.newaxis]  # shape (3, N)
        
        # Project onto tangent vector to get signed distance along tangent
        r = np.dot(self.tangent, dx)  # shape (N,)
        
        # Compute parabolic profile factor (zero outside radius)
        factor = np.where(np.abs(r) <= self.radius, 1 - (r / self.radius) ** 2, 0.0)
        
        velocity_magnitude = self.v_max * factor  # shape (N,)
        
        # Velocity is along normal direction
        velocity = self.normal[:, np.newaxis] * velocity_magnitude[np.newaxis, :]

        dim = self.mesh.get_gdim()
        if dim == 2:
            # In 2D, return only x and y components
            return velocity[:2, :]
        
        return velocity


class PlugInletProfile:
    def __init__(self, speed, normal, center=None, radius=None):
        """
        Uniform or circular plug inlet profile in 3D.

        Parameters
        ----------
        speed : float
            Uniform inlet speed (magnitude of velocity).
        normal : array_like
            Inlet normal vector (3D, not necessarily unit length).
        center : array_like, optional
            Center of the circular inlet (3D). Required if radius is given.
        radius : float, optional
            Radius of the circular inlet. If None, velocity is uniform everywhere.
        """
        self.speed = speed
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)  # unit normal
        self.center = np.array(center, dtype=float) if center is not None else None
        self.radius = radius

        if self.radius is not None and self.center is None:
            raise ValueError("Must provide center if radius is specified")

    def update_speed(self, new_speed):
        """Update inlet speed."""
        self.speed = new_speed

    def __call__(self, x):
        """
        Evaluate velocity profile at points x

        Parameters
        ----------
        x : np.ndarray of shape (3, num_points)

        Returns
        -------
        np.ndarray of shape (3, num_points)
            Velocity vectors
        """

        dx = x - self.center[:, np.newaxis] if self.center is not None else x
        r = np.linalg.norm(dx, axis=0) 
        
        if self.radius is None:
            factor = np.ones(dx.shape[1])
        else:
            factor = np.where(r <= self.radius, 1.0, 0.0)

        velocity_magnitude = self.speed * factor

        return self.normal[:, np.newaxis] * velocity_magnitude

