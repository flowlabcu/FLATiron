import dolfinx
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

class LinearSolver:
    """
    A solver for physics problems whose weak form is exactly linear in the
    solution function. Does a single assemble + single PETSc KSP solve --
    no Newton/SNES loop, no re-linearization, no convergence-check
    iteration. Analogous to NonLinearSolver, but for problems where that
    machinery is unnecessary overhead.

    Parameters
    --------------
    comm (MPI.Comm):
        The MPI communicator.
    problem (LinearProblem):
        The linear problem to solve.
    **kwargs:
        Arbitrary keyword arguments to configure the solver.

        Common kwargs include:

            - outer_ksp_set_function (callable): A function to customize the KSP solver.
                                                It should take one argument: the PETSc KSP object.
    """

    def __init__(self, comm: MPI.Comm, problem, **kwargs):
        self._mpi_comm = comm
        self.problem = problem
        self.ksp_is_initialized = False

        self._outer_ksp_set_func = kwargs.pop("outer_ksp_set_function", self.default_set_ksp)

        self.krylov_solver = PETSc.KSP().create(comm)

        self._A = None
        self._b = None

        self.init_ksp()

    def init_ksp(self):
        """
        Applies the KSP/PC setup function once. No matrix/vector assembly
        happens here -- that's done in assemble(), so the problem is only
        ever assembled once per solve().
        """
        if self.ksp_is_initialized:
            return

        self._outer_ksp_set_func(self.krylov_solver)
        self.ksp_is_initialized = True

    def set_ksp_option(self, ksp: PETSc.KSP, keyword: str, value):
        """
        Helper method to set a PETSc KSP option using its option prefix.

        Parameters
        ------------
        ksp (PETSc.KSP):
            The KSP object.
        keyword (str):
            The PETSc option keyword (e.g., "ksp_type", "pc_type").
        value:
            The value for the option.
        """
        prefix = ksp.getOptionsPrefix()
        opts = PETSc.Options()
        opts[f"{prefix}{keyword}"] = value

    def default_set_ksp(self, ksp: PETSc.KSP):
        """
        Sets default values for the KSP solver, used if no custom KSP setup
        function is provided by the user. Mirrors NonLinearSolver's default
        (robust direct solve); problem-specific solvers (e.g. CG + AMG for
        an SPD Poisson system) should be passed in via
        outer_ksp_set_function.

        Parameters
        ----------
        ksp (PETSc.KSP):
            The KSP object to configure.
        """
        self.set_ksp_option(ksp, 'ksp_type', 'gmres')
        self.set_ksp_option(ksp, 'pc_type', 'lu')
        self.set_ksp_option(ksp, 'pc_factor_mat_solver_type', 'mumps')
        self.set_ksp_option(ksp, 'ksp_rtol', 1e-9)
        self.set_ksp_option(ksp, 'ksp_atol', 1e-12)
        self.set_ksp_option(ksp, 'ksp_max_it', 500)
        ksp.setFromOptions()

    def assemble(self):
        """
        Assemble the linear system (matrix + RHS) exactly once and attach
        it to the KSP. Exposed as its own method (rather than folded
        silently into solve()) so callers can time assembly separately
        from the KSP solve if needed.

        Returns
        -------
        (PETSc.Mat, PETSc.Vec)
            The assembled system matrix and right-hand-side vector.
        """
        jac_form = dolfinx.fem.form(self.problem.jacobian)
        res_form = dolfinx.fem.form(self.problem.weak_form)

        A = fem_petsc.assemble_matrix(jac_form, bcs=self.problem.physics.dirichlet_bcs)
        A.assemble()

        b = fem_petsc.create_vector(res_form)
        with b.localForm() as b_local:
            b_local.set(0.0)
        fem_petsc.assemble_vector(b, res_form)
        fem_petsc.apply_lifting(b, [jac_form], bcs=[self.problem.physics.dirichlet_bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, self.problem.physics.dirichlet_bcs)

        self.krylov_solver.setOperators(A)
        self._A = A
        self._b = b
        return A, b

    def solve(self):
        """
        Assembles the linear system once and solves it once.
        """
        self.init_ksp()
        self.assemble()

        self.krylov_solver.solve(self._b, self.problem.physics.solution.x.petsc_vec)
        self.problem.physics.solution.x.scatter_forward()

        num_iterations = self.krylov_solver.getIterationNumber()
        converged = self.krylov_solver.getConvergedReason() > 0
        if self._mpi_comm.rank == 0:
            print(f"Linear solver converged in {num_iterations} iterations.")

        return num_iterations, converged
