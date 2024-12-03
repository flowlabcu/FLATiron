import pytest
from flatiron_tk.physics import *
from flatiron_tk.mesh import BoxMesh, RectMesh, LineMesh
import fenics as fe
from flatiron_tk.solver import PhysicsSolver
import numpy as np


def test_dofs(mesh_3d, vector_equal):

    stk = StokesFlow(mesh_3d)
    stk.set_element('CG', 1, 'CG', 1)

    st0 = ScalarTransport(mesh_3d, tag='c0')
    st0.set_element('CG', 1)

    st1 = ScalarTransport(mesh_3d, tag='c1')
    st1.set_element('CG', 1)

    mphys = MultiPhysicsProblem(st0, st1, stk)
    mphys.set_element()
    mphys.build_function_space()

    dofs_st0 = mphys.V.sub(0).dofmap().dofs() 
    assert vector_equal(mphys.get_dofs('c0'), dofs_st0)

    dofs_st1 = mphys.V.sub(1).dofmap().dofs() 
    assert vector_equal(mphys.get_dofs('c1'), dofs_st1)

    dofs_stk = mphys.V.sub(2).dofmap().dofs()
    dofs_stk_u = mphys.V.sub(2).sub(0).dofmap().dofs()
    dofs_stk_p = mphys.V.sub(2).sub(1).dofmap().dofs()
    assert vector_equal(mphys.get_dofs('u'), dofs_stk_u)
    assert vector_equal(mphys.get_dofs('p'), dofs_stk_p)
    assert vector_equal(mphys.get_dofs( ('u','p') ), dofs_stk)
