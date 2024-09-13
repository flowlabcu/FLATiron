
// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/ts/CVode.h>

namespace py = pybind11;

namespace dolfin_wrappers
{

  void ts(py::module& m)
  {

    #ifdef HAS_SUNDIALS
    // CVode trampoline class for allowing overloading of virtual functions
    class PyCVode : public dolfin::CVode
    {
    public:
      using dolfin::CVode::CVode;

      void derivs(double t, std::shared_ptr<dolfin::GenericVector> u,
                  std::shared_ptr<dolfin::GenericVector> udot)
      { PYBIND11_OVERLOAD_NAME(void, dolfin::CVode, "derivs", derivs,
                                t, u, udot); }

      int jacobian(std::shared_ptr<const dolfin::GenericVector> v,
                   std::shared_ptr<dolfin::GenericVector> Jv,
                   double t,
                   std::shared_ptr<const dolfin::GenericVector> y,
                   std::shared_ptr<const dolfin::GenericVector> fy)
      { PYBIND11_OVERLOAD_NAME(int, dolfin::CVode, "jacobian", jacobian,
                                v, Jv, t, y, fy); }

      int psolve(double t,
                 std::shared_ptr<const dolfin::GenericVector> y,
                 std::shared_ptr<const dolfin::GenericVector> fy,
                 std::shared_ptr<const dolfin::GenericVector> r,
                 std::shared_ptr<dolfin::GenericVector> z,
                 double gamma, double delta, int lr)
      { PYBIND11_OVERLOAD_NAME(int, dolfin::CVode, "psolve", psolve,
                               t, y, fy, r, z, gamma, delta, lr); }
    };

    py::class_<dolfin::CVode, PyCVode, std::shared_ptr<dolfin::CVode>>cvode(m,"CVode");
    //dolfin::CVode
    cvode
      .def(py::init<dolfin::CVode::LMM, dolfin::CVode::ITER>())
      .def("init", &dolfin::CVode::init, py::arg("u0"), py::arg("atol"),
        py::arg("rtol"), py::arg("mxsteps")=0)
      .def("set_time", &dolfin::CVode::set_time, py::arg("t0"))
      .def("get_time", &dolfin::CVode::get_time)
      .def("step", &dolfin::CVode::step, py::arg("dt"))
      .def("statistics", &dolfin::CVode::statistics)
      .def("derivs", &dolfin::CVode::derivs)
      .def("jacobian", &dolfin::CVode::jacobian)
      .def("psolve", &dolfin::CVode::psolve);

    py::enum_<dolfin::CVode::LMM>(cvode,"LMM")
      .value("CV_BDF", dolfin::CVode::LMM::cv_bdf)
      .value("CV_ADAMS", dolfin::CVode::LMM::cv_adams);

    py::enum_<dolfin::CVode::ITER>(cvode,"ITER")
      .value("CV_FUNCTIONAL", dolfin::CVode::ITER::cv_functional)
      .value("CV_NEWTON", dolfin::CVode::ITER::cv_newton);

    #endif

  }
}
