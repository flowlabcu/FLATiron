import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

from dolfin import PETScMatrix, Timer, DirichletBC
from petsc4py import PETSc

class PCDAssembler(object):
    """Base class for creating linear problems to be solved by application
    of the PCD preconditioning strategy. Users are encouraged to use this class
    for interfacing with :py:class:`fenapack.field_split.PCDKrylovSolver`.
    On request it assembles not only the individual PCD operators but also the
    system matrix and the right hand side vector defining the linear problem.
    """

    def __init__(self, a, L, bcs, a_pc=None,
                 mp=None, mu=None, ap=None, fp=None, kp=None, gp=None,
                 bcs_pcd=[]):
        """Collect individual variational forms and boundary conditions
        defining a linear problem (system matrix + RHS vector) on the one side
        and preconditioning operators on the other side.

        *Arguments*
            a (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing a system matrix.
            L (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Linear form representing a right hand side vector.
            bcs (:py:class:`list` of :py:class:`dolfin.DirichletBC`)
                Boundary conditions applied to ``a``, ``L``, and ``a_pc``.
            a_pc (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing a matrix optionally passed to
                preconditioner instead of ``a``. In case of PCD, stabilized
                00-block can be passed to 00-KSP solver.
            mp, mu, ap, fp, kp, gp (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear forms which (some of them) might be used by a
                particular PCD(R) preconditioner. Typically they represent "mass
                matrix" on pressure, "mass matrix" on velocity, minus Laplacian
                operator on pressure, pressure convection-diffusion operator,
                pressure convection operator and pressure gradient respectively.
            bcs_pcd (:py:class:`list` of :py:class:`dolfin.DirichletBC`)
                Artificial boundary conditions used by PCD preconditioner.

        All the arguments should be given on the common mixed function space.

        All the forms are wrapped using :py:class:`PCDForm` so that each of
        them can be endowed with additional set of properties.

        By default, ``mp``, ``mu``, ``ap`` and ``gp`` are assumed to be
        constant if the preconditioner is used repeatedly in some outer
        iterative process (e.g Newton-Raphson method, time-stepping).
        As such, the corresponding operators are assembled only once.
        On the other hand, ``fp`` and ``kp`` are updated in every
        outer iteration.

        Also note that ``gp`` is the only form that is by default in a *phantom
        mode*. It means that the corresponding operator (if needed) is not
        obtained by assembling the form, but it is extracted as the 01-block of
        the system matrix.

        The default setting can be modified by accessing
        a :py:class:`PCDForm` instance via :py:meth:`PCDAssembler.get_pcd_form`
        and changing the properties directly.
        """

        # Assembler for the linear system of algebraic equations
        self.assembler = SystemAssembler(a, L, bcs)

        # Assembler for preconditioner
        if a_pc is not None:
            self.assembler_pc = SystemAssembler(a_pc, L, bcs)
        else:
            self.assembler_pc = None

        # Store bcs
        self._bcs = bcs
        self._bcs_pcd = bcs_pcd

        # Store and initialize forms
        self._forms = {
            "L": PCDForm(L),
            "ap": PCDForm(ap, const=True),
            "mp": PCDForm(mp, const=True),
            "mu": PCDForm(mu, const=True),
            "fp": PCDForm(fp),
            "kp": PCDForm(kp),
            "gp": PCDForm(gp, const=True, phantom=True),
        }


    def get_pcd_form(self, key):
        """Return form wrapped in :py:class:`PCDForm`."""
        form = self._forms.get(key)
        if form is None:
            raise AttributeError("Form '%s' requested by PCD not available" % key)
        assert isinstance(form, PCDForm)
        return form


    def get_dolfin_form(self, key):
        """Return form as :py:class:`dolfin.Form` or :py:class:`ufl.Form`."""
        return self.get_pcd_form(key).dolfin_form()


    def function_space(self):
        return self.get_dolfin_form("L").arguments()[0].function_space()


    def rhs_vector(self, b, x=None):
        """Assemble right hand side vector ``b``.

        The version with ``x`` is suitable for use inside
        a (quasi)-Newton solver.
        """
        if x is not None:
            self.assembler.assemble(b, x)
        else:
            self.assembler.assemble(b)

    def system_matrix(self, A):
        """Assemble system matrix ``A``."""
        self.assembler.assemble(A)


    def pc_matrix(self, P):
        """Assemble preconditioning matrix ``P`` whose relevant blocks can be
        passed to actual parts of the ``KSP`` solver.
        """
        if self.assembler_pc is not None:
            self.assembler_pc.assemble(P)


    def ap(self, Ap):
        assembler = SystemAssembler(self.get_dolfin_form("ap"),
                                    self.get_dolfin_form("L"),
                                    self.pcd_bcs())
        assembler.assemble(Ap)


    def mp(self, Mp):
        assemble(self.get_dolfin_form("mp"), tensor=Mp)


    def mu(self, Mu):
        assemble(self.get_dolfin_form("mu"), tensor=Mu)


    def fp(self, Fp):
        assemble(self.get_dolfin_form("fp"), tensor=Fp)


    def kp(self, Kp):
        assemble(self.get_dolfin_form("kp"), tensor=Kp)


    def gp(self, Bt):
        """Assemble discrete pressure gradient. It is crucial to respect any
        constraints placed on the velocity test space by Dirichlet boundary
        conditions."""
        assemble(self.get_dolfin_form("gp"), tensor=Bt)
        for bc in self._bcs:
            bc.apply(Bt)


    # FIXME: Naming
    def pcd_bcs(self):
        try:
            assert self._bcs_pcd is not None
        except (AttributeError, AssertionError):
            raise AttributeError("BCs requested by PCD not available")
        return self._bcs_pcd



class PCDInterface(object):
    """Wrapper of PCDAssembler for interfacing with PCD PC
    fieldsplit implementation. Convection fieldsplit submatrices
    are extracted as shallow or deep submatrices according to
    ``deep_submats`` parameter."""

    def __init__(self, pcd_assembler, A, is_u, is_p, deep_submats=False):
        """Create PCDInterface instance given PCDAssembler instance,
        system matrix and velocity and pressure index sets"""

        # Check input
        assert isinstance(pcd_assembler, PCDAssembler)
        assert isinstance(is_u, PETSc.IS)
        assert isinstance(is_p, PETSc.IS)

        # Store what needed
        self.assembler = pcd_assembler
        self.A = proxy(A)
        self.is_u = is_u
        self.is_p = is_p

        # Choose submatrix implementation
        assert isinstance(deep_submats, bool)
        if deep_submats:
            self.assemble_operator = self._assemble_operator_deep
        else:
            self.assemble_operator = self._assemble_operator_shallow

        # Dictionary for storing work mats
        self.scratch = WeakKeyDictionary()


    def apply_pcd_bcs(self, vec):
        """Apply bcs to intermediate pressure vector of PCD pc"""
        self.apply_bcs(vec, self.assembler.pcd_bcs, self.is_p)


    def setup_ksp_Ap(self, ksp):
        """Setup pressure Laplacian ksp and assemble matrix"""
        self.setup_ksp(ksp, self.assembler.ap, self.is_p, spd=True,
                       const=self.assembler.get_pcd_form("ap").is_constant())


    def setup_ksp_Mp(self, ksp):
        """Setup pressure mass matrix ksp and assemble matrix"""
        self.setup_ksp(ksp, self.assembler.mp, self.is_p, spd=True,
                       const=self.assembler.get_pcd_form("mp").is_constant())


    def setup_mat_Kp(self, mat=None):
        """Setup and assemble pressure convection
        matrix and return it"""
        if mat is None or not self.assembler.get_pcd_form("kp").is_constant():
            return self.assemble_operator(self.assembler.kp, self.is_p, submat=mat)


    def setup_mat_Fp(self, mat=None):
        """Setup and assemble pressure convection-diffusion
        matrix and return it"""
        if mat is None or not self.assembler.get_pcd_form("fp").is_constant():
            return self.assemble_operator(self.assembler.fp, self.is_p, submat=mat)


    def setup_mat_Mu(self, mat=None):
        """Setup and assemble velocity mass matrix
        and return it"""
        # NOTE: deep submats are required for the later use in _build_approx_Ap
        if mat is None or not self.assembler.get_pcd_form("mu").is_constant():
            return self._assemble_operator_deep(self.assembler.mu, self.is_u, submat=mat)


    def setup_mat_Bt(self, mat=None):
        """Setup and assemble discrete pressure gradient
        and return it"""
        # NOTE: deep submats are required for the later use in _build_approx_Ap
        if mat is None or not self.assembler.get_pcd_form("gp").is_constant():
            if self.assembler.get_pcd_form("gp").is_phantom():
                # NOTE: Bt is obtained from the system matrix
                return self._get_deep_submat(self.A, self.is_u, self.is_p, submat=mat)
            else:
                # NOTE: Bt is obtained by assembling a form
                return self._assemble_operator_deep(self.assembler.gp,
                                                    self.is_u, self.is_p, submat=mat)


    def setup_ksp_Rp(self, ksp, Mu, Bt):
        """Setup pressure Laplacian ksp based on velocity mass matrix ``Mu``
        and discrete gradient ``Bt`` and assemble matrix
        """
        mat = ksp.getOperators()[0]
        prefix = ksp.getOptionsPrefix()
        const = self.assembler.get_pcd_form("mu").is_constant() \
                  and self.assembler.get_pcd_form("gp").is_constant()
        if mat.type is None or not mat.isAssembled() or not const:
            # Get approximate Laplacian
            mat = self._build_approx_Ap(Mu, Bt, mat)

            # Use eventual spd flag
            mat.setOption(PETSc.Mat.Option.SPD, True)

            # Set correct options prefix
            mat.setOptionsPrefix(prefix)

            # Use also as preconditioner matrix
            ksp.setOperators(mat, mat)
            assert ksp.getOperators()[0].isAssembled()

            # Setup ksp
            with Timer("FENaPack: {} setup".format(prefix)):
                ksp.setUp()


    def _build_approx_Ap(self, Mu, Bt, mat=None):
        # Fetch work vector and matrix
        diagMu, = self.get_work_vecs_from_square_mat(Mu, 1)
        Ap, = self.get_work_mats(Bt, 1)

        # Get diagonal of the velocity mass matrix
        Mu.getDiagonal(result=diagMu)

        # Make inverse of diag(Mu)
        diagMu.reciprocal() # diag(Mu)^{-1}

        # Make square root of the diagonal and use it for scaling
        diagMu.sqrtabs() # \sqrt{diag(Mu)^{-1}}

        # Process discrete "grad" operator
        Bt.copy(result=Ap)         # Ap = Bt
        Ap.diagonalScale(L=diagMu) # scale rows of Ap, i.e. Ap = diagMu*Bt

        # Return Ap = Ap^T*Ap, which is B diag(Mu)^{-1} B^T,
        if mat is None or not mat.isAssembled():
            return Ap.transposeMatMult(Ap)
        else:
            # NOTE: 'result' can only be used if the multiplied matrices have
            #       the same nonzero pattern as in the previous call
            return Ap.transposeMatMult(Ap, result=mat)


    def get_work_vecs_from_square_mat(self, M, num):
        """Return ``num`` of work vecs initially created from a square
        matrix ``M``."""
        # Verify that we have a square matrix
        m, n = M.getSize()
        assert m == n
        try:
            vecs = self._work_vecs
            assert len(vecs) == num
        except AttributeError:
            self._work_vecs = vecs = tuple(M.getVecLeft() for i in range(num))
        except AssertionError:
            raise ValueError("Changing number of work vecs not allowed")
        return vecs


    def get_work_mats(self, M, num):
        """Return ``num`` of work mats initially created from matrix ``B``."""
        try:
            mats = self._work_mats
            assert len(mats) == num
        except AttributeError:
            self._work_mats = mats = tuple(M.duplicate() for i in range(num))
        except AssertionError:
            raise ValueError("Changing number of work mats not allowed")
        return mats


    def get_work_dolfin_mat(self, key, comm,
                            can_be_destroyed=None, can_be_shared=None):
        """Get working DOLFIN matrix by key. ``can_be_destroyed=True`` tells
        that it is probably favourable to not store the matrix unless it is
        shared as it will not be used ever again, ``None`` means that it can
        be destroyed but it is not probably favourable and ``False`` forbids
        the destruction. ``can_be_shared`` tells if a work matrix can be the
        same with work matrices for other keys."""
        # TODO: Add mechanism for sharing DOLFIN mats
        # NOTE: Maybe we don't really need sharing. If only persistent matrix
        #       is convection then there is nothing to be shared.

        # Check if requested matrix is in scratch
        dolfin_mat = self.scratch.get(key, None)

        # Allocate new matrix otherwise
        if dolfin_mat is None:

            if isinstance(comm, PETSc.Comm):
                comm = comm.tompi4py()

            dolfin_mat = PETScMatrix(comm)

        # Store or pop the matrix as requested
        if can_be_destroyed in [False, None]:
            self.scratch[key] = dolfin_mat
        else:
            assert can_be_destroyed is True
            self.scratch.pop(key, None)

        return dolfin_mat


    def setup_ksp(self, ksp, assemble_func, iset, spd=False, const=False):
        """Assemble into operator of given ksp if not yet assembled"""
        mat = ksp.getOperators()[0]
        prefix = ksp.getOptionsPrefix()
        if mat.type is None or not mat.isAssembled():
            # Assemble matrix
            destruction = True if const else None
            dolfin_mat = self.get_work_dolfin_mat(assemble_func, mat.comm,
                                                  can_be_destroyed=destruction,
                                                  can_be_shared=True)
            assemble_func(dolfin_mat)
            mat = self._get_deep_submat(dolfin_mat.mat(), iset, submat=None)

            # Use eventual spd flag
            mat.setOption(PETSc.Mat.Option.SPD, spd)

            # Set correct options prefix
            mat.setOptionsPrefix(prefix)

            # Use also as preconditioner matrix
            ksp.setOperators(mat, mat)
            assert ksp.getOperators()[0].isAssembled()

            # Set up ksp
            with Timer("FENaPack: {} setup".format(prefix)):
                ksp.setUp()

        elif not const:
            # Assemble matrix and set up ksp
            mat = self._assemble_operator_deep(assemble_func, iset, submat=mat)
            assert mat.getOptionsPrefix() == prefix
            ksp.setOperators(mat, mat)
            with Timer("FENaPack: {} setup".format(prefix)):
                ksp.setUp()


    def _assemble_operator_shallow(self, assemble_func, isrow, iscol=None, submat=None):
        """Assemble operator of given name using shallow submat"""
        # Assemble into persistent DOLFIN matrix everytime
        # TODO: Does not shallow submat take care of parents lifetime? How?
        dolfin_mat = self.get_work_dolfin_mat(assemble_func, isrow.comm,
                                              can_be_destroyed=False,
                                              can_be_shared=False)
        assemble_func(dolfin_mat)

        # FIXME: This logic that it is created once should be visible
        #        in higher level, not in these internals
        # Create shallow submatrix (view into dolfin mat) once
        if submat is None or submat.type is None or not submat.isAssembled():
            submat = self._get_shallow_submat(dolfin_mat.mat(), isrow, iscol, submat=submat)
            assert submat.isAssembled()

        return submat


    def _assemble_operator_deep(self, assemble_func, isrow, iscol=None, submat=None):
        """Assemble operator of given name using deep submat"""
        dolfin_mat = self.get_work_dolfin_mat(assemble_func, isrow.comm,
                                              can_be_destroyed=None,
                                              can_be_shared=True)
        assemble_func(dolfin_mat)
        return self._get_deep_submat(dolfin_mat.mat(), isrow, iscol, submat=submat)


    def apply_bcs(self, vec, bcs_getter, iset):
        """Transform dolfin bcs obtained using ``bcs_getter`` function
        into fieldsplit subBCs and apply them to fieldsplit vector.
        SubBCs are cached."""
        # Fetch subbcs from cache or construct it
        subbcs = getattr(self, "_subbcs", None)
        if subbcs is None:
            bcs = bcs_getter()
            bcs = [bcs] if isinstance(bcs, DirichletBC) else bcs
            subbcs = [SubfieldBC(bc, iset) for bc in bcs]
            self._subbcs = subbcs

        # Apply bcs
        for bc in subbcs:
            bc.apply(vec)


    if PETSc.Sys.getVersion()[0:2] <= (3, 7) and PETSc.Sys.getVersionInfo()['release']:

        @staticmethod
        def _get_deep_submat(mat, isrow, iscol=None, submat=None):
            if iscol is None:
                iscol = isrow
            return mat.getSubMatrix(isrow, iscol, submat=submat)

        @staticmethod
        def _get_shallow_submat(mat, isrow, iscol=None, submat=None):
            if iscol is None:
                iscol = isrow
            if submat is None:
                submat = PETSc.Mat().create(isrow.comm)
            return submat.createSubMatrix(mat, isrow, iscol)


    else:

        @staticmethod
        def _get_deep_submat(mat, isrow, iscol=None, submat=None):
            if iscol is None:
                iscol = isrow
            return mat.createSubMatrix(isrow, iscol, submat=submat)

        @staticmethod
        def _get_shallow_submat(mat, isrow, iscol=None, submat=None):
            if iscol is None:
                iscol = isrow
            if submat is None:
                submat = PETSc.Mat().create(isrow.comm)
            return submat.createSubMatrixVirtual(mat, isrow, iscol)

