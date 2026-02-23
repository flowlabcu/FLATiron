import atexit
import dolfinx
import flatiron_tk
import numpy as np
import pyvista as pv
import os
import shutil

from flatiron_tk import PointEvaluator
from flatiron_tk import PVDWriter
from mpi4py import MPI

def _ensure_3D_velocity(velocity):
    """
    Convert 1D/2D/3D velocity into a length-3 numpy array.
    """
    velocity = np.array(velocity, dtype=np.float64).ravel()
    if velocity.size < 3:
        velocity = np.pad(velocity, (0, 3 - velocity.size), mode='constant')
    elif velocity.size > 3:
        raise ValueError('Velocity evaluation vector must be 1D, 2D, or 3D.')
    return velocity

class MasslessTracerTracker:
    """
    Track massless particles and write per-step .vtp files plus a .pvd collection
    (finalized automatically via atexit). Works under MPI: only rank 0 writes the .pvd,
    while all ranks may write .vtp files if desired.

    Parameters
    ----------
    mesh : Mesh
        The flatirion_tk mesh object.
    dt : float
        Time step size for particle advection.
    """

    def __init__(self, mesh, dt):
        self.mesh = mesh
        self.dt = float(dt)
        
        self.V = dolfinx.fem.functionspace(self.mesh.msh, ('CG', 1))
        self.evaluator = PointEvaluator(mesh)
        
        # Particle positions are nx3 numpy array
        self.particle_positions = np.zeros((0, 3), dtype=np.float64)
        self.step = 0
        self._finalized = False

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Register automatic finalizer (safe to register multiple times)
        atexit.register(self._finalize_if_needed)

    def set_writer(self, output_dir):
        """
        Prepare output folder and PVD writer.
        Parameters
        ----------
        output_dir : str
            Directory to write per-step files and particles.pvd.
        """
        self.file_format = 'pvd'
        self.output_dir = os.path.abspath(output_dir)

        # Only rank 0 should create/remove the directory, then synchronize
        if self.rank == 0:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

        # wait until output_dir exists on shared FS
        self.comm.Barrier()

        # Setup PVD only on rank 0
        if self.rank == 0:
            pvd_path = os.path.join(self.output_dir, 'particles.pvd')
            self.pvd_writer = PVDWriter(pvd_path)
        

    def write(self, time_stamp=None):
        """
        Write current particle positions to a .vtp file and add to a .pvd collection.
        Parameters
        ----------
        time_stamp : float
            Time stamp to associate with this write in the .pvd file. 
        """

        positions = np.asarray(self.particle_positions, dtype=np.float64)
        positions_file = os.path.join(self.output_dir, f'positions_step_p{self.rank:04d}_{self.step:08d}.vtp')

        # Ensure nx3
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if positions.shape[1] == 2:
            positions = np.hstack((positions, np.zeros((positions.shape[0], 1), dtype=positions.dtype)))

        points = pv.PolyData(positions)
        if positions.shape[0] > 0:
            points["Particle ID"] = np.arange(positions.shape[0], dtype=np.int64)

        # Save file
        try:
            points.save(positions_file)

        except Exception as e:
            if self.rank == 0:
                flatiron_tk.custom_warning_message(f'Failed to save VTP {positions_file}: {e}')

        # Only rank 0 adds the dataset entry to the PVD
        if self.rank == 0 and self.pvd_writer is not None:
            timestep_value = time_stamp if time_stamp is not None else self.step
            for r in range(self.size):
                file_for_rank = f"positions_step_p{r:04d}_{self.step:08d}.vtp"
                self.pvd_writer.add_dataset(timestep=timestep_value, file=file_for_rank)

        self.step += 1
        self.comm.Barrier()
        self.step += 1
        # ensure PVD updates are visible after write (rank 0 only)
        self.comm.Barrier()

    def set_particle_positions(self, particle_positions):
        """
        Set the particle positions directly.
        Parameters
        ----------
        particle_positions : Sequence
            Iterable of particle positions, each a 1D array-like of length 2 or 3.
        """
        arr = np.array(particle_positions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == 2:
            arr = np.hstack((arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)))
        self.particle_positions = arr

    def set_particle_positions_from_boundary(self, boundary_id):
        """
        Set the particle positions to all DOF coordinates on a given boundary.
        Parameters
        ----------
        boundary_id : int
            The boundary marker ID from which to get the coordinates.
        """
        self.particle_positions = self.get_coordinates_on_boundary(boundary_id)

    def get_coordinates_on_boundary(self, boundary_id):
        """
        Get the coordinates of all DOFs on a given boundary across all ranks.
        Parameters
        ----------
        boundary_id : int
            The boundary marker ID from which to get the coordinates.
        Returns
        -------
        coords : np.ndarray
            Nx3 array of coordinates on the given boundary across all ranks.
        """
        dofs_on_boundary = dolfinx.fem.locate_dofs_topological(self.V, self.mesh.get_fdim(), self.mesh.boundary.find(boundary_id))
        dof_coords = self.V.tabulate_dof_coordinates()
        coords_on_boundary = dof_coords[dofs_on_boundary, :]

        # Make nx3
        if coords_on_boundary.shape[1] == 2:
            coords_on_boundary = np.hstack(
                (coords_on_boundary, np.zeros((coords_on_boundary.shape[0], 1)))
            )

        coords_on_boundary = coords_on_boundary.astype(np.float64)

        # Gather all coordinates
        all_coords = self.comm.allgather(coords_on_boundary)
        all_coords = np.vstack(all_coords) if any(len(c) > 0 for c in all_coords) else np.zeros((0, 3), dtype=np.float64)

        if all_coords.shape[0] == 0:
            raise ValueError(f'No coordinates found on boundary with marker {boundary_id} across all ranks.')

        return all_coords

    def update_particle_positions(self, current_velocity, previous_velocity=None, method='euler'):
        """
        Update particle positions based on the current and previous velocity fields.
        Parameters
        ----------
        current_velocity : dolfinx.fem.Function
            The current velocity field.
        previous_velocity : dolfinx.fem.Function
            The previous velocity field (required for Heun's method).
        method : str
            The time integration method to use ('euler' or 'heun').
        """
        un = current_velocity
        u0 = previous_velocity

        if method == "euler":
            _, vel_n = self.evaluator.evaluate_set(un, self.particle_positions)
            for pos, vel in zip(self.particle_positions, vel_n):
                if vel is not None:
                    vel = _ensure_3D_velocity(vel)
                    pos += vel * self.dt
                

        elif method == "heun":
            if u0 is None:
                raise ValueError('previous velocity must be provided for Heun\'s method.')

            _, vel_0 = self.evaluator.evaluate_set(u0, self.particle_positions)

            for pos, vel0 in zip(self.particle_positions, vel_0):
                if vel0 is not None:
                    vel0 = _ensure_3D_velocity(vel0)
                    predictor_pos = pos + vel0 * self.dt
                    vel_pred = self.evaluator.evaluate_point(un, predictor_pos, show_warning=False)
                    
                    if vel_pred is None:
                        vel_pred = np.zeros(3)
                    else:
                        vel_pred = _ensure_3D_velocity(vel_pred)

                    pos += 0.5 * (vel0 + vel_pred) * self.dt

    def inject_particles(self, new_particle_positions):
        """
        Inject new particles at specified positions.
        Parameters
        ----------
        new_particle_positions : Sequence
            Iterable of new particle positions, each a 1D array-like of length 2 or 3.
        """
        arr = np.array(new_particle_positions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == 2:
            arr = np.hstack((arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)))
        if self.particle_positions.size == 0:
            self.particle_positions = arr
        else:
            self.particle_positions = np.vstack((self.particle_positions, arr))

    def inject_particles_from_boundary(self, boundary_id):
        boundary_coords = self.get_coordinates_on_boundary(boundary_id)
        self.inject_particles(boundary_coords)

    def _finalize_if_needed(self):
        """
        Automatically called at program exit to write the .pvd file on rank 0.
        """
        try:
            if getattr(self, "_finalized", False):
                return

            if not hasattr(self, 'file_format') or self.file_format not in ('vtp', 'pvd'):
                # nothing to do
                self._finalized = True
                return

            # only rank 0 writes the pvd
            if self.rank == 0 and getattr(self, 'pvd_writer', None) is not None:
                try:
                    self.pvd_writer.write()
                except Exception as e:
                    flatiron_tk.custom_warning_message(f'Failed to finalize particle PVD file: {e}')
            # mark finalized on all ranks
            self._finalized = True
        
        except Exception:
            # Never fail on exit hooks
            pass