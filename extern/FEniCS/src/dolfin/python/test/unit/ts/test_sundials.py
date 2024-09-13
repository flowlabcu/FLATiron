#!/usr/bin/python3

import numpy
from dolfin import *
from dolfin_utils.test import skip_if_not_SUNDIALS


@skip_if_not_SUNDIALS
def test_sundials_adams():

    class MyCVode(CVode):

        def derivs(self, t, u, udot):
            udot[:] = -u[:]

        def jacobian(self, v, Jv, t, y, fy):
            Jv[:] = v[:]
            return 0

        def psolve(self, t, u, udot, r, z, gamma, x, y):
            z[:] = r[:]
            return 0

    phi = Vector(MPI.comm_world, 10)
    phi[:] = 1.0

    cv = MyCVode(CVode.LMM.CV_ADAMS, CVode.ITER.CV_FUNCTIONAL)
    cv.init(phi, 1e-7, 1e-7)

    nstep = 200
    dt = 0.01
    for i in range(nstep):
        t = cv.step(dt)
        assert (exp(-t) - phi[0]) < 1e-6

@skip_if_not_SUNDIALS
def test_sundials_newton():

    class MyCVode(CVode):

        def derivs(self, t, u, udot):
            udot[:] = -u[:]

        def jacobian(self, v, Jv, t, y, fy):
            Jv[:] = v[:]
            return 0

        def psolve(self, t, u, udot, r, z, gamma, x, y):
            z[:] = r[:]
            return 0

    phi = Vector(MPI.comm_world, 10)
    phi[:] = 1.0

    cv = MyCVode(CVode.LMM.CV_BDF, CVode.ITER.CV_NEWTON)
    cv.init(phi, 1e-7, 1e-7)

    nstep = 200
    dt = 0.01
    for i in range(nstep):
        t = cv.step(dt)
        assert (exp(-t) - phi[0]) < 1e-6

@skip_if_not_SUNDIALS
def test_sundials_diffusion_1d():
    # Finite difference test
    mesh = UnitIntervalMesh(MPI.comm_world, 100)
    Q = FunctionSpace(mesh, "CG", 1)
    F = Function(Q)
    b = 2.0
    F.interpolate(Expression("exp(-b*pow(x[0] - 0.5, 2))", b=b, degree=1))
    phi = F.vector()

    class tmp_test(CVode):
        def derivs(self, t, phi, phidot):
            p = phi[:]
            pd = numpy.zeros_like(p)
            for i in range(1, len(p)-1):
                pd[i] = p[i-1] - 2*p[i] + p[i+1]
            phidot[:] = pd

        def jacobian(self, v, Jv, t, y, fy):
            Jv[:] = v[:]
            return 0
        def psolve(self, t, u, udot, r, z, gamma, x, y):
            z[:] = r[:]
            return 0

    cv = tmp_test(CVode.LMM.CV_BDF, CVode.ITER.CV_NEWTON)
    cv.init(phi, 1e-7, 1e-7)

    t0 = 1.0/(4.0*b)
    cv.set_time(t0)

    nstep = 200
    dt = 0.25
    for i in range(nstep):
        t = cv.step(dt)
        print(t, phi.max(), phi.sum(), sqrt(t)*phi.max())
