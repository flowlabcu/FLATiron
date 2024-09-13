// Copyright (C) 2017 Chris Richardson and Chris Hadjigeorgiou
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
//

#ifdef HAS_SUNDIALS

#include <cmath>
#include <vector>
#include <iostream>
#include <map>
#include <string>

#include <dolfin/common/MPI.h>
#include <dolfin/la/SUNDIALSNVector.h>
#include <dolfin/log/log.h>

#include <cvode/cvode.h>
#include <cvode/cvode_bandpre.h>
#include <cvode/cvode_impl.h>
#include <cvode/cvode_spils.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_dense.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_iterative.h>



#include "CVode.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CVode::CVode(LMM cv_lmm, ITER cv_iter) : _ls(NULL), _t(0.0), _cv_iter(cv_iter)
{
  // Create CVode memory block
  _cvode_mem = CVodeCreate(cv_lmm, cv_iter);
  dolfin_assert(_cvode_mem);

  // Point user_data back to this object
  // (for use in "f" below)
  int flag = CVodeSetUserData(_cvode_mem, (void *)this);
  dolfin_assert(flag == 0);
}
//-----------------------------------------------------------------------------
CVode::~CVode()
{
  CVodeFree(&_cvode_mem);
  if (_ls)
    SUNLinSolFree(_ls);
}
//-----------------------------------------------------------------------------
void CVode::init(std::shared_ptr<GenericVector> u0, double atol, double rtol, long int mxsteps)
{
  dolfin_assert(_cvode_mem);

  // Make a sundials n_vector sharing data with u0
  _u = std::make_shared<SUNDIALSNVector>(u0);

  // Initialise
  int flag = CVodeInit(_cvode_mem, f, _t, _u->nvector());
  dolfin_assert(flag == CV_SUCCESS);

  flag = CVodeSStolerances(_cvode_mem, rtol, atol);
  dolfin_assert(flag == CV_SUCCESS);

  CVodeSetMaxNumSteps(_cvode_mem, mxsteps);

  if (_cv_iter == CV_NEWTON)
  {
    dolfin_debug("Initialising Newton solver");
    _ls = SUNSPGMR(_u->nvector(), PREC_LEFT, 0);
    flag = CVSpilsSetLinearSolver(_cvode_mem, _ls);
    dolfin_assert(flag == CV_SUCCESS);

    // Set the preconditioner solver function to be called by CVode solver
    flag = CVSpilsSetPreconditioner(_cvode_mem, NULL, prec_solve);
    dolfin_assert(flag == CV_SUCCESS);

    // Set the Jacobian function to be called by CVode solver
    flag = CVSpilsSetJacTimes(_cvode_mem, NULL, f_jac);
    dolfin_assert(flag == CV_SUCCESS);
  }

}
//-----------------------------------------------------------------------------
double CVode::step(double dt)
{
  double tout = _t + dt;
  int flag = ::CVode(_cvode_mem, tout, _u->nvector(), &_t, CV_NORMAL);
  dolfin_assert((flag == CV_SUCCESS) || (flag == CV_TSTOP_RETURN) || (flag == CV_ROOT_RETURN));

  return _t;
}
//-----------------------------------------------------------------------------
double CVode::get_time() const
{
  return _t;
}
//-----------------------------------------------------------------------------
void CVode::set_time(double t0)
{
  _t = t0;
}
//-----------------------------------------------------------------------------
void CVode::derivs(double t, std::shared_ptr<GenericVector> u,
                   std::shared_ptr<GenericVector> udot)
{
  dolfin_error("CVode.cpp",
               "form time derivative",
               "This function should be overloaded");
}
//-----------------------------------------------------------------------------
int CVode::jacobian(std::shared_ptr<const GenericVector> v,
                    std::shared_ptr<GenericVector> Jv,
                    double t, std::shared_ptr<const GenericVector> y,
                    std::shared_ptr<const GenericVector> fy)
{
  dolfin_error("CVode.cpp",
      	       "compute Jacobian function",
               "This function should be overloaded");
  return 0;
}
//-----------------------------------------------------------------------------
int CVode::jacobian_setup(double t,
                          std::shared_ptr<GenericVector> Jv,
                          std::shared_ptr<GenericVector> y)
{
  dolfin_error("CVode.cpp",
      	       "set up Jacobian function",
               "This function should be overloaded");
  return 0;
}

//-----------------------------------------------------------------------------
int CVode::psolve(double t, std::shared_ptr<const GenericVector>y,
                  std::shared_ptr<const GenericVector> fy,
                  std::shared_ptr<const GenericVector> r,
                  std::shared_ptr<GenericVector> z,
                  double gamma, double delta, int lr)
{
  /// Overloaded preconditioner solver function
  return 0;
}

//-----------------------------------------------------------------------------
int CVode::f_jac_setup(double t, N_Vector y, N_Vector fy, void *user_data)
{

  CVode* cv = static_cast<CVode*>(user_data);

  auto yvec = static_cast<const SUNDIALSNVector*>(y->content)->vec();
  auto fyvec = static_cast<SUNDIALSNVector*>(fy->content)->vec();

  cv->jacobian_setup(t, yvec, fyvec);
  return 0;
}
//-----------------------------------------------------------------------------
int CVode::f_jac(N_Vector v, N_Vector Jv, double t, N_Vector y, N_Vector fy, void* user_data, N_Vector tmp)
{

  CVode* cv = static_cast<CVode*>(user_data);

  auto vvec = static_cast<const SUNDIALSNVector*>(v->content)->vec();
  auto Jvvec = static_cast<SUNDIALSNVector*>(Jv->content)->vec();

  auto yvec = static_cast<const SUNDIALSNVector*>(y->content)->vec();
  auto fyvec = static_cast<SUNDIALSNVector*>(fy->content)->vec();
  auto tmpvec = static_cast<SUNDIALSNVector*>(tmp->content)->vec();

  cv->jacobian(vvec, Jvvec, t, yvec, fyvec);

  return 0;
}
//-----------------------------------------------------------------------------
int CVode::f(realtype t, N_Vector u, N_Vector udot, void *user_data)
{
  // f is a static function (from C), so need to get pointer to "this" object
  // passed though in user_data
  CVode* cv = static_cast<CVode*>(user_data);

  auto uvec = static_cast<const SUNDIALSNVector*>(u->content)->vec();
  auto udotvec = static_cast<SUNDIALSNVector*>(udot->content)->vec();

  // Callback to actually calculate the derivatives (user function)
  cv->derivs(t, uvec, udotvec);

  return 0;
}

//--------------------------------------------------------------------------
int CVode::prec_solve(double t, N_Vector y, N_Vector fy, N_Vector r, N_Vector z,
                      double gamma, double delta, int lr, void *user_data)
{
  // Preconditioner solve routine
  // TODO: Create as virtual function

  CVode* cv = static_cast<CVode*>(user_data);

  auto yvec = static_cast<const SUNDIALSNVector*>(y->content)->vec();
  auto fyvec = static_cast<const SUNDIALSNVector*>(fy->content)->vec();
  auto rvec = static_cast<const SUNDIALSNVector*>(r->content)->vec();
  auto zvec = static_cast<SUNDIALSNVector*>(z->content)->vec();

  cv->psolve(t, yvec, fyvec, rvec, zvec, gamma, delta, lr);

  return 0;
}
//-----------------------------------------------------------------------------
std::map<std::string, double> CVode::statistics()
{
  std::map<std::string, double> stats;
  auto cv = static_cast<const CVodeMem>(_cvode_mem);

  stats["Steps"] = cv->cv_nst;
  stats["RHSEvals"] = cv->cv_nfe;
  stats["LinSolvSetups"] = cv->cv_nsetups;
  stats["ErrTestFails"] = cv->cv_netf;
  stats["LastOrder"] = cv->cv_qu;
  stats["CurrentOrder"] = cv->cv_next_q;
  stats["StabLimOrderReds"] = cv->cv_nor;
  stats["ActualInitStep"] = cv->cv_h0u;
  stats["LastStep"] = cv->cv_hu;
  stats["CurrentStep"] = cv->cv_next_h;
  stats["CurrentTime"] = cv->cv_tn;
  stats["TolScaleFactor"] = cv->cv_tolsf;
  stats["NumGEvals"] = cv->cv_nge;
  stats["NumNonlinSolvIter"] = cv->cv_nni;
  stats["NumNonlinSolvConvFails"] = cv->cv_ncfn;

  return stats;
}
#endif
