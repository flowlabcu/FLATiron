import numpy as np

from petsc4py import PETSc

class ConvergenceMonitor():
    """
    Object which reports and records the convergence history of an iterative method.
    This is meant to be used with PETSc's iterative method objects.

    Parameters
        ----------
        name : str
            Name to identify the iterative method (e.g., 'ksp', 'snes',
            'ts').
        verbose : bool, optional
            If True, prints the iteration number and residual norm to
            standard output every `report_every` iterations. Default is True.
        report_every : int, optional
            Frequency of reporting the iteration number and residual norm.
            Default is 1 (every iteration).
        record_history : bool, optional
            If True, records the convergence history (iteration numbers and
            residual norms). Default is True.
        comm : MPI.Comm, optional
            MPI communicator for parallel execution. Default is PETSc.COMM_WORLD.   
    """

    def __init__(self, name, verbose=True, report_every=1, record_history=True, comm=PETSc.COMM_WORLD):
        self._verbose = verbose
        self._name = name
        self._report_every = report_every
        self._record_history = record_history
        self._comm = comm
        self.reset_convergence_history()

    def __call__(self, method, it, rnorm):
        """
        This method is called by PETSc's iterative methods at each iteration.
        
        Parameters
        ----------
        method : PETSc.KSP, PETSc.SNES, or PETSc.TS
            The iterative method object.
        it : int
            The current iteration number.
        rnorm : float
            The current residual norm.
        """
        # Need the method input here for it to work with PETSc's object
        if self._record_history:
            self.rnorm.append(rnorm)
            self.it.append(it)

        if self._verbose and self._comm.rank==0 and it%self._report_every==0:
            print("%s iteration: %5d  rnorm: %2.15e"%(self._name, it, rnorm))

    def convergence_history(self):
        """
        Returns the recorded convergence history as a list of arrays.
        Each array corresponds to a segment of the convergence history where
        the iteration count was monotonically increasing.
        """
        
        # Find indices where the iteration count decreases (i.e., a new segment starts)
        new_it = []
        for i in range(1, len(self.it)):
            if self.it[i] < self.it[i-1]:
                new_it.append(i)
        istart = 0
        # Convert recorded lists to numpy arrays for slicing
        self.rnorm = np.array(self.rnorm)
        self.it = np.array(self.it)
        rnorms = []
        # Split the residual norms into segments based on new_it indices
        for i in new_it:
            rnorms.append(self.rnorm[istart:i])
            istart = i
        # Add the last segment
        rnorms.append(self.rnorm[istart:])
        return rnorms

    def reset_convergence_history(self):
        """
        Resets the recorded convergence history.
        """

        self.rnorm = []
        self.it = []