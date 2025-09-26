^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Block Solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function builds a block preconditioner P for ksp solve of matrix A::

    Let ksp(A, P) means a Krylov solve of matrix A with preconditioner P

    Define a block matrix system A as

        A = [ A00, A01
              A10, A11 ]

    Composite type options are:

        additive: P = [ksp(A00,Ap00), 0
                       0            , ksp(A11,Ap11)]

        multiplicative: P = J.K.L
            where J = [I, 0
                       0, ksp(A11,Ap11)]

                  K = [0, 0    +   [ I  , 0     *[I, 0        (see PETSc's doc page for details here)
                       0, I]        -A10, -A11]   0, 0]

                  L = [ksp(A00,Ap00), 0
                       0            , I]

        symmetric_multiplicative: (see PETSc's doc page. This is too big)

        schur: (schur complement decomposition, see doc page)

.. automodule:: flatiron_tk.solver.block_solver
   :members:
   :undoc-members:
   :show-inheritance:
