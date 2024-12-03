# from petsc4py import PETSc
from ..info.messages import import_PETSc
PETSc = import_PETSc()

class ConvergenceMonitor():
    pass

if PETSc is not None:

    class ConvergenceMonitor():

        """
        Object which reports and records the convergence history of an iterative method.
        This is meant to be used with PETSc's iterative method objects
        """

        def __init__(self,
                     name,
                     verbose=True,
                     report_every=1,
                     record_history=True,
                     comm=PETSc.COMM_WORLD):

            self._verbose = verbose
            self._name = name
            self._report_every = report_every
            self._record_history = record_history
            self._comm = comm
            self.reset_convergence_history()

        def __call__(self, method, it, rnorm):

            # Need the method input here for it to work with PETSc's object

            if self._record_history:
                self.rnorm.append(rnorm)
                self.it.append(it)

            if self._verbose and self._comm.rank==0 and it%self._report_every==0:
                print("%s iteration: %5d  rnorm: %2.15e"%(self._name, it, rnorm))

        def convergence_history(self):

            new_it = []
            for i in range(1, len(self.it)):
                if self.it[i] < self.it[i-1]:
                    new_it.append(i)
            istart = 0
            self.rnorm = np.array(self.rnorm)
            self.it = np.array(self.it)
            rnorms = []
            for i in new_it:
                rnorms.append(self.rnorm[istart:i])
                istart = i
            rnorms.append(self.rnorm[istart:])
            return rnorms

        def reset_convergence_history(self):

            self.rnorm = []
            self.it = []


