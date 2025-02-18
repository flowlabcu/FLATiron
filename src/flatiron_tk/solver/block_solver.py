import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os

# ------------------------------------------------------- #

from collections.abc import Iterable
from ..info.messages import import_PETSc
PETSc = import_PETSc()
# from petsc4py import PETSc
from .convergence_monitor import ConvergenceMonitor
from .non_linear_problem import NonLinearSolver


"""
This function builds a block preconditioner P for ksp solve of matrix A.

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
"""

def _is_container(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


class FieldSplitNode():

    def __init__(self, left=None, right=None):

        self.left = left
        self.right = right
        self.parent = None
        self._is_root = False
        self._fields = set()
        self.tag = ''
        self._ksp = None
        self._monitor = None

    def sorted_fields(self):
        _fields = copy.deepcopy(list(self.fields()))
        _fields.sort()
        return tuple(_fields)

    def set_ksp(self, ksp, _is=None):
        self._ksp = ksp
        if _is is not None:
            self._is = _is

    def ksp(self):
        return self._ksp

    def set_fields(self, fields):
        self._fields = fields
        self.tag = '_'.join(fields)

    def fields(self):
        '''
        Return true if this node contains every input fields exactly
        '''
        return self._fields

    def _set_monitor(self, monitor):
        self._monitor = monitor

    def monitor(self):
        return self._monitor

    def _insert_overwrite(self, child_node, left_or_right):
        assert (left_or_right == 'left' or left_or_right == 'right')
        if left_or_right == 'left':
            self.left = child_node
        else:
            self.right = child_node
        child_node.parent = self

    def insert(self, node):
        assert (self.left is None or self.right is None)
        if self.left is None:
            self._insert_overwrite(node, 'left')
        else:
            self._insert_overwrite(node, 'right')

    def is_root(self):
        return self._is_root

    def set_as_root(self):
        self._is_root = True

class FieldSplitTree():

    def __init__(self, physics, splits):

        self.physics = physics
        self.root = FieldSplitNode()
        self.root.set_fields(list(physics.tag.keys()))
        self.root.set_as_root()
        self.node_dict = {}
        self.node_dict[self.root.sorted_fields()] = self.root

        # Store the splits as a list of dicts
        # I have this if else here to handle the
        # case where we have a single split, so
        # the user can just supply the split dict
        if isinstance(splits, dict):
            self.splits = [splits]
        else:
            self.splits = splits

        # Build dictionary for PETSc fieldsplit inputs
        # NOTE: This step can probably be automated for each dict,
        # but I am doing it explicitly here so it's easy to
        # see what's going on.

        # Field split composite type. This tells you what the final
        # preconditioner look like
        # See https://petsc.org/release/manual/ksp/#sec-block-matrices
        self._composite_type_dict = {}
        self._composite_type_dict['additive'] = PETSc.PC.CompositeType.ADDITIVE
        self._composite_type_dict['multiplicative'] = PETSc.PC.CompositeType.MULTIPLICATIVE
        self._composite_type_dict['schur'] = PETSc.PC.CompositeType.SCHUR
        self._composite_type_dict['special'] = PETSc.PC.CompositeType.SPECIAL
        self._composite_type_dict['symmetric_multiplicative'] = PETSc.PC.CompositeType.SYMMETRIC_MULTIPLICATIVE

        # Schur complement factorization type
        # If composite_type is SCHUR, this tells you
        # what the final preconditioner involving the schur complement
        # look like.
        # See: https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/
        self._schur_fact_type_dict = {}
        self._schur_fact_type_dict['diag'] = PETSc.PC.SchurFactType.DIAG
        self._schur_fact_type_dict['full'] = PETSc.PC.SchurFactType.FULL
        self._schur_fact_type_dict['lower'] = PETSc.PC.SchurFactType.LOWER
        self._schur_fact_type_dict['upper'] = PETSc.PC.SchurFactType.UPPER

        # Schur complement preconditioner type
        # Preconditioner for the schur complement block
        # See https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/
        self._schur_pre_type_dict = {}
        self._schur_pre_type_dict['a11'] = PETSc.PC.SchurPreType.A11
        self._schur_pre_type_dict['full'] = PETSc.PC.SchurPreType.FULL
        self._schur_pre_type_dict['self'] = PETSc.PC.SchurPreType.SELF
        self._schur_pre_type_dict['selfp'] = PETSc.PC.SchurPreType.SELFP
        self._schur_pre_type_dict['user'] = PETSc.PC.SchurPreType.USER

    def set_root_ksp(self, root_ksp):
        self.root.set_ksp(root_ksp)

    # def split(self, node, fields_0, fields_1, **split_settings):
    def split(self, fields_0, fields_1, **split_settings):

        """
        **split_settings:
             composite_type
             schur_pre_type
             schur_fact_type
        """

        # Make the input a list for consistent implementation of this function
        _fields_0, _fields_1 = fields_0, fields_1
        if not _is_container(_fields_0):
            _fields_0 = [_fields_0]
        if not _is_container(_fields_1):
            _fields_1 = [_fields_1]

        # Make sure all fields in the parent node are present in field_0 and field_1
        parent_fields = list(_fields_0) + list(_fields_1)
        parent_fields.sort()
        parent_fields = tuple(parent_fields)
        assert parent_fields in self.node_dict
        parent_node = self.node_dict[parent_fields]
        # assert(set(node.fields()) == set(all_fields))


        # Get fieldsplit name as a combination of the supplied fields
        # delimited by `_`
        field_0_name = '_'.join(list(_fields_0))
        field_1_name = '_'.join(list(_fields_1))

        # Set left and right node
        lnode = FieldSplitNode()
        lnode.set_fields(_fields_0)
        parent_node.insert(lnode)
        self.node_dict[lnode.sorted_fields()] = lnode

        rnode = FieldSplitNode()
        rnode.set_fields(_fields_1)
        parent_node.insert(rnode)
        self.node_dict[rnode.sorted_fields()] = rnode

        # Finally build the fieldsplit index set
        dofs0 = self.physics.get_dofs(parent_node.left.fields(), sort=False)
        dofs1 = self.physics.get_dofs(parent_node.right.fields(), sort=False)
        is0, is1 = self.get_fieldsplit_IS(parent_node, dofs0, dofs1)
        ksp0, ksp1 = self._build_fieldsplit_pc(parent_node.ksp(),
                                               is0, field_0_name,
                                               is1, field_1_name,
                                               **split_settings)
        parent_node.left.set_ksp(ksp0, is0)
        parent_node.right.set_ksp(ksp1, is1)
        return ksp0, ksp1

    def get_fieldsplit_IS(self, node, dofs0, dofs1):

        """
        Get index set for dofs0 and dofs1 relative to the current node.
        """

        if node.is_root():
            is0 = PETSc.IS().createGeneral(dofs0).sort()
            is1 = PETSc.IS().createGeneral(dofs1).sort()
            return is0, is1

        comm = self.physics.mesh.comm
        ndofs0 = len(dofs0)
        ndofs1 = len(dofs1)
        node_dofs = np.array(dofs0+dofs1, dtype=np.int32)
        sorted_ids = np.argsort(node_dofs)

        sub_ids_0 = np.where(sorted_ids<ndofs0)[0].astype(np.int32)
        sub_ids_1 = np.where(sorted_ids>=ndofs0)[0].astype(np.int32)
        ndofs_sub = ndofs0 + ndofs1
        offset = np.cumsum([0]+comm.allgather(ndofs_sub))[comm.rank]

        sub_ids_0 += offset
        sub_ids_1 += offset
        is0 = PETSc.IS().createGeneral(sub_ids_0).sort()
        is1 = PETSc.IS().createGeneral(sub_ids_1).sort()

        return is0, is1

    def _build_fieldsplit_pc(self, outer_ksp,
                             is0, is0_name,
                             is1, is1_name,
                             composite_type='additive',
                             schur_pre_type='a11',
                             schur_fact_type='full'):

        """
        Here I am explicitly building a petsc dictionary
        """

        # Build fieldsplit preconditioner with the supplied index set
        pc = outer_ksp.pc
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        pc.setFieldSplitIS([is0_name, is0], [is1_name, is1])

        # Set composite type
        assert composite_type in self._composite_type_dict
        comp_type = self._composite_type_dict[composite_type]
        pc.setFieldSplitType(comp_type)

        # If composite type is schur, set schur complement settings
        if composite_type == 'schur':

            # Set schur preconditioner type
            assert schur_pre_type in self._schur_pre_type_dict
            pre_type = self._schur_pre_type_dict[schur_pre_type]
            pc.setFieldSplitSchurPreType(pre_type)

            # Set factorization type
            assert schur_fact_type in self._schur_fact_type_dict
            fact_type = self._schur_fact_type_dict[schur_fact_type]
            pc.setFieldSplitSchurFactType(fact_type)

        outer_ksp.setUp()

        return pc.getFieldSplitSubKSP()

class BlockNonLinearSolver(NonLinearSolver):

    def __init__(self, fieldsplit_tree, *args, **kwargs):
        self._fs_tree = fieldsplit_tree
        super().__init__(*args, **kwargs)

    def init_ksp(self):

        # Set the outer ksp solver
        ksp = self.linear_solver().ksp()
        self._outer_ksp_set_func(ksp)

        # Set inner solvers
        # By default I define set_ksp0 and set_ksp1 
        # here to rename the monitor
        self._fs_tree.root.set_ksp(ksp)
        for split in self._fs_tree.splits:
            fields = split.pop('fields')
            ksp0_set_function = split.pop('ksp0_set_function', self.default_set_ksp0)
            ksp1_set_function = split.pop('ksp1_set_function', self.default_set_ksp1)
            ksp0, ksp1 = self._fs_tree.split(fields[0], fields[1], **split)
            ksp0_set_function(ksp0)
            ksp1_set_function(ksp1)
        ksp.setUp()

    def default_set_ksp0(self, ksp):
        super().default_set_ksp(ksp)
        ksp.setMonitor(ConvergenceMonitor('ksp0'))

    def default_set_ksp1(self, ksp):
        super().default_set_ksp(ksp)
        ksp.setMonitor(ConvergenceMonitor('ksp1'))

def main():

    from flatiron_tk.physics import PhysicsProblem, MultiPhysicsProblem
    from flatiron_tk.mesh import LineMesh
    from flatiron_tk.solver import NonLinearProblem, NonLinearSolver
    import fenics as fe

    class GalerkinProjection(PhysicsProblem):
        '''
        GalerkinProjection field_value = b
        '''
        def set_element(self, element_family, element_degree, dim):
            self.element = fe.VectorElement(element_family, self.mesh.mesh.ufl_cell(), element_degree, dim=dim)
            self.element_family = element_family
            self.element_degree = element_degree

        def set_projection_value(self, projection_value):
            self.set_external_function('b', projection_value)

        def flux(self):
            ''''''

        def get_residue(self):
            ''''''

        def set_weak_form(self):
            b = self.external_function('b')
            u = self.solution_function()
            w = self.test_function()
            self.weak_form = fe.dot(u-b, w)*self.dx

    def build_GP(tag, mesh, dim, val):
        GP = GalerkinProjection(mesh, tag)
        GP.set_element('CG', 1, dim)
        GP.set_projection_value(fe.Constant(val))
        return GP

    mesh = LineMesh(0, 1, 1/10)
    GP1 = build_GP('A', mesh, dim=2, val=[1,2])
    GP2 = build_GP('B', mesh, dim=3, val=[3,4,5])
    GP3 = build_GP('C', mesh, dim=4, val=[6,7,8,9])
    GPs = [GP1, GP2, GP3]

    physics = MultiPhysicsProblem(*GPs)
    physics.set_element()
    physics.build_function_space()
    physics.set_weak_form()

    tree = FieldSplitTree(physics)

    split0 = {'fields': (('A','C'),'B'),
              'composite_type': 'schur',
              'schur_fact_type': 'full',
              'schur_pre_type': 'a11'}

    split1 = {'fields': ('A','C'),
              'composite_type': 'schur',
              'schur_fact_type': 'full',
              'schur_pre_type': 'a11'}

    splits = [split0, split1]



    problem = NonLinearProblem(physics)
    solver = BlockNonLinearSolver(tree, splits, fe.MPI.comm_world, problem, fe.PETScKrylovSolver())
    solver.solve()
    (A, B, C) = physics.solution.split(True)
    print('A', A.vector()[:])
    print('B', B.vector()[:])
    print('C', C.vector()[:])

if __name__ == '__main__':
    main()


