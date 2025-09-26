import dolfinx
from .convergence_monitor import ConvergenceMonitor
from dolfinx.fem import petsc as fem_petsc # Import fem.petsc for NonlinearProblem
from dolfinx.nls import petsc as nls_petsc # Alias nls.petsc to avoid name collision
from mpi4py import MPI
from petsc4py import PETSc

class NonLinearSolver(nls_petsc.NewtonSolver): 
    """
    A wrapper class around dolfinx.nls.petsc.NewtonSolver for solving
    nonlinear PDE problems within the flatironx framework.

    This class handles the setup of the nonlinear solver's convergence
    criteria and provides a mechanism to customize the underlying
    linear (KSP) solver's settings.
    
    Parameters
    --------------
    comm (MPI.Comm): 
        The MPI communicator.
    problem (dolfinx.fem.petsc.NonlinearProblem): 
        The nonlinear problem to solve.
    **kwargs: 
        Arbitrary keyword arguments to configure the solver.

        Common kwargs include:
        
            - atol (float): Absolute tolerance for the nonlinear solver.
            - rtol (float): Relative tolerance for the nonlinear solver.
            - report (bool): Whether to report convergence.
            - relaxation_parameter (float): Relaxation parameter for Newton's method.
            - max_it (int): Maximum number of nonlinear iterations.
            - convergence_criterion (str): Convergence criterion ("incremental" or "residual").
            - outer_ksp_set_function (callable): A function to customize the KSP solver.
                                                It should take one argument: the PETSc KSP object.
            - post_ksp_setup_hook (callable): A hook function to run after KSP setup. 
    """

    def __init__(self, comm: MPI.Comm, problem: fem_petsc.NonlinearProblem, **kwargs):
        self._mpi_comm = comm
        self.problem = problem
        self.ksp_is_initialized = False # Flag to ensure KSP setup is done once

        # Determine the function to set KSP options.
        # If 'outer_ksp_set_function' is provided, use it; otherwise, use default_set_ksp.
        self._outer_ksp_set_func = kwargs.pop("outer_ksp_set_function", self.default_set_ksp)
        
        # Initialize the base NewtonSolver class
        super().__init__(comm, problem)

        # Set NewtonSolver's own properties from kwargs or defaults
        # These attributes are set directly on the NonLinearSolver instance
        self._Atol = kwargs.get('atol', 1e-10) 
        self.rtol = kwargs.get('rtol', 1e-7)  
        self.report = kwargs.get('report', True) 
        self.relaxation_parameter = kwargs.get('relaxation_parameter', 1.0) 
        self.max_it = kwargs.get('max_it', 100) 
        self.convergence_criterion = kwargs.get('convergence_criterion', "incremental")
        self._post_ksp_setup_hook = kwargs.get('post_ksp_setup_hook', None)
        

        self.init_ksp()


    def init_ksp(self):
        """
        Initializes the KSP solver, sets matrix structure
        and applies post-setup hooks like FieldSplit configuration.
        """
        if self.ksp_is_initialized:
            return

        # Outer KSP setup (basic type, tolerances)
        self._outer_ksp_set_func(self.krylov_solver)

        # Assemble matrix once to define sparsity structure
        jac_form = dolfinx.fem.form(self.problem.jacobian)
        self._A = dolfinx.fem.petsc.assemble_matrix(jac_form, bcs=self.problem.physics.dirichlet_bcs)
        self._A.assemble()

        # Preallocate b with the right layout
        res_form = dolfinx.fem.form(self.problem.weak_form)
        self._b = dolfinx.fem.petsc.create_vector(res_form)

        self.krylov_solver.setOperators(self._A)
        self.krylov_solver.setUp()

        # Call the post-setup hook for nested splits, IS definitions, etc.
        if self._post_ksp_setup_hook:
            self._post_ksp_setup_hook(self.krylov_solver)

        self.ksp_is_initialized = True

    def set_ksp_option(self, ksp: PETSc.KSP, keyword: str, value):
        """
        Helper method to set a PETSc KSP option using its option prefix.
        This ensures options are specific to this KSP instance.

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
        Sets default values for the KSP solver. This method is used if no
        custom KSP setup function is provided by the user.
        It uses `set_ksp_option` and `ksp.setFromOptions()` for consistency.

        Parameters
        ----------
        ksp (PETSc.KSP): 
            The KSP object to configure.
        """
        # Set common robust defaults for linear solves in nonlinear problems
        self.set_ksp_option(ksp, 'ksp_type', 'gmres')
        self.set_ksp_option(ksp, 'pc_type', 'lu')
        self.set_ksp_option(ksp, 'pc_factor_mat_solver_type', 'mumps') # Mumps is highly recommended for mixed problems
        
        # Set tolerances for the linear solver
        self.set_ksp_option(ksp, 'ksp_rtol', 1e-9)
        self.set_ksp_option(ksp, 'ksp_atol', 1e-12)
        self.set_ksp_option(ksp, 'ksp_max_it', 500) # Max iterations for the linear solver

        # Apply all options set via PETSc.Options()
        ksp.setFromOptions()
        
        # Set a monitor for the linear solver's convergence
        ksp.setMonitor(ConvergenceMonitor('ksp'))

    
    def solve(self):
        """
        Solves the nonlinear problem using the Newton method.
        """
        # Ensure KSP initialized (returns early if already done)
        self.init_ksp()

        # Assemble matrix and RHS
        jac_form = dolfinx.fem.form(self.problem.jacobian)
        res_form = dolfinx.fem.form(self.problem.weak_form)

        self._A.zeroEntries()  # Clear previous entries
        dolfinx.fem.petsc.assemble_matrix(self._A, jac_form, bcs=self.problem.physics.dirichlet_bcs)
        self._A.assemble()

        with self._b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self._b, res_form)
        dolfinx.fem.petsc.apply_lifting(self._b, [jac_form], bcs=[self.problem.physics.dirichlet_bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(self._b, self.problem.physics.dirichlet_bcs)


        # Reattach new A matrix
        self.krylov_solver.setOperators(self._A)

        # Call nonlinear solver
        num_iterations, converged = super().solve(self.problem.physics.solution)
        if self._mpi_comm.rank == 0:
            print(f"Nonlinear solver converged in {num_iterations} iterations.")
            
        return num_iterations, converged
