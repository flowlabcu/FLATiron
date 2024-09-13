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

#ifndef __DOLFIN_C_VODE_H
#define __DOLFIN_C_VODE_H

#ifdef HAS_SUNDIALS

#include <dolfin/la/SUNDIALSNVector.h>

#include <cvode/cvode.h>
#include <cvode/cvode_impl.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_dense.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_iterative.h>

namespace dolfin
{
  /// Wrapper class to SUNDIALS CVODE
  class CVode
  {
  public:

    // These enums are used by PYBIND11 to map the definitions from C
    enum LMM { cv_bdf = CV_BDF, cv_adams = CV_ADAMS };

    enum ITER { cv_functional = CV_FUNCTIONAL, cv_newton = CV_NEWTON };

    /// Constructor
    /// @param cv_lmm
    ///   linear multistep method
    /// @param cv_iter
    ///   iteration type
    CVode(LMM cv_lmm, ITER cv_iter);

    /// Destructor
    virtual ~CVode();

    /// Initialise CVode
    /// @param u0
    ///   Input vector
    /// @param atol
    ///   absolute tolerance
    /// @param rtol
    ///   relative tolerance
    /// @param mxsteps
    ///   maximum number of steps
    void init(std::shared_ptr<GenericVector> u0, double atol, double rtol, long int mxsteps = 0);

    /// Advance time by timestep dt
    /// @param dt
    ///   timestep 
    /// @return
    ///   CVODE return flag
    double step(double dt);

    /// Get current time
    /// @return double
    ///   current time
    double get_time() const;

    /// Set the current time
    /// @param t0
    ///   current time
    void set_time(double t0);

    /// Overloaded function for time derivatives of u at time t.
    /// Given the vector u, at time t, provide the time derivative udot.
    /// @param t
    ///   time
    /// @param u
    ///   input vector of values u
    /// @param udot
    ///   output vector containing computed derivative of u at time t
    virtual void derivs(double t, std::shared_ptr<GenericVector> u,
                        std::shared_ptr<GenericVector> udot);

    /// Given the values (t, y, fy, v), compute Jv = (df/dy)v
    /// @param v
    ///   vector to be multiplied by the Jacobian df/dy 
    /// @param Jv
    ///   output vector of (df/dy)*v
    /// @param t
    ///   current value of the independent variable.
    /// @param y
    ///   current value of the ependent variable.
    /// @param fy
    ///   vector f(t,y)
    /// @return
    ///   success flag, 0 if successful
    virtual int jacobian(std::shared_ptr<const GenericVector> v,
                         std::shared_ptr<GenericVector> Jv,
                         double t, std::shared_ptr<const GenericVector> y,
                         std::shared_ptr<const GenericVector> fy);

    /// User-defined setup function called once per Newton iteration.
    /// Data structures for usage by the Jacobian function can be setup here
    /// @param t
    ///   current value of the independent variable 
    /// @param Jv
    ///   current value of the dependent variable vector,
    ///   namely the predicted value of y(t).
    /// @param y
    ///   vector f(t,y). 
    /// @return
    ///   success flag, 0 if success
    virtual int jacobian_setup(double t,
                               std::shared_ptr<GenericVector> Jv,
                               std::shared_ptr<GenericVector> y);

    /// Overloaded preconditioner solver function
    /// @param tn
    ///   current value of the independent variable.
    /// @param y
    ///   current value of the dependent variable vector.
    /// @param fy
    ///   vector f(t,y) 
    /// @param r
    ///   right-hand side vector of the linear system.
    /// @param z
    ///   output vector computed by PrecSolve.
    /// @param gamma
    ///   scalar appearing in the Newton matrix.
    /// @param delta
    ///   input tolerance if an iterative method is used.
    /// @param lr
    ///   input flag indicating whether to use left or right preconditioner.
    /// @return
    ///   success flag, 0 if success 
    virtual int psolve(double tn, std::shared_ptr<const GenericVector>y,
                       std::shared_ptr<const GenericVector> fy,
                       std::shared_ptr<const GenericVector> r,
                       std::shared_ptr<GenericVector> z,
                       double gamma, double delta, int lr);

    /// Return statistics
    /// @return
    ///   map structure containing information stored in the CVode
    ///   structure, ie. number of solver steps, RHS evaluations, current time
    std::map<std::string,double> statistics();

  private:
    /// Internal callback from CVode to get time derivatives
    /// Executes the overloaded derivs function(above)
    static int f(realtype t, N_Vector u, N_Vector udot, void *user_data);

    /// Internal callback from CVode to perform user-defined setup before calling Jacobian function
    /// Executes the overloaded jacobian_setup function (above)
    static int f_jac_setup(double t, N_Vector y, N_Vector fy, void *user_data);

    /// Internal callback from CVode to the Jacobian estimation function
    /// Executes the overloaded jacobian function (above)
    static int f_jac(N_Vector u, N_Vector fu, double t, N_Vector y, N_Vector fy, void* , N_Vector tmp);

    /// Internal callback from CVode to the preconditioner solver function
    /// Executes the overloaded psolve function (above)
    static int prec_solve(double, N_Vector, N_Vector, N_Vector, N_Vector, double, double, int, void*);

    // Vector of values - wrapper around dolfin::GenericVector
    std::shared_ptr<SUNDIALSNVector> _u;

    // SUNDIALS Linear Solver
    SUNLinearSolver _ls;

    // Current time
    double _t;

    // Pointer to CVode memory struct
    void* _cvode_mem;

    // Remember iter method between constructor and init
    ITER _cv_iter;
  };

}

#endif

#endif
