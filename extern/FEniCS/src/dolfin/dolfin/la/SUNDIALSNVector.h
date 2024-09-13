// Copyright (C) 2017 Chris Hadjigeorgiou and Chris Richardson
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


#ifndef __DOLFIN_N_VECTOR_H
#define __DOLFIN_N_VECTOR_H

#ifdef HAS_SUNDIALS

#include <string>
#include <utility>
#include <memory>
#include <dolfin/common/types.h>
#include <sundials/sundials_nvector.h>
#include "DefaultFactory.h"
#include "GenericVector.h"
#include "Vector.h"

namespace dolfin
{
  ///
  ///   Interface to SUNDIALS NVector
  ///
  class SUNDIALSNVector
  {
  public:

    /// Create empty vector
    /// @param comm
    ///    MPI communicator
    SUNDIALSNVector(MPI_Comm comm=MPI_COMM_WORLD)
    {
      DefaultFactory factory;
      vector = factory.create_vector(comm);
    }

    /// Create vector of size N
    /// @param comm
    ///    MPI communicator
    /// @param N
    ///    Size of vector
    SUNDIALSNVector(MPI_Comm comm, std::size_t N)
    {
      DefaultFactory factory;
      vector = factory.create_vector(comm);
      vector->init(N);
      N_V = std::unique_ptr<_generic_N_Vector>(new _generic_N_Vector);
      N_V->ops = &ops;
      N_V->content = (void *)(this);
    }

    /// Copy constructor
    /// @param x
    ///    SUNDIALSNVector to copy
    SUNDIALSNVector(const SUNDIALSNVector& x) : vector(x.vec()->copy()) {}

    /// Create an SUNDIALSNVector from a GenericVector
    /// @param x
    ///    GenericVector to copy
    SUNDIALSNVector(const GenericVector& x) : vector(x.copy())
    {
      N_V = std::unique_ptr<_generic_N_Vector>(new _generic_N_Vector);
      N_V->ops = &ops;
      N_V->content = (void *)(this);
    }

    /// Create a SUNDIALSNVector wrapper to an existing GenericVector
    /// @param x
    ///    GenericVector pointer to copy
    SUNDIALSNVector(std::shared_ptr<GenericVector> x) : vector(x)
    {
      N_V = std::unique_ptr<_generic_N_Vector>(new _generic_N_Vector);
      N_V->ops = &ops;
      N_V->content = (void *)(this);
    }
    //-----------------------------------------------------------------------------

    /// Get underlying raw SUNDIALS N_Vector struct
    /// @return 
    ///   raw SUNDIALS N_Vector struct
    N_Vector nvector() const
    {
      N_V->content = (void *)(this);
      return N_V.get();
    }

    /// Get underlying GenericVector
    /// @return 
    ///   underlying GenericVector
    std::shared_ptr<GenericVector> vec() const
    {
      return vector;
    }

    /// Assignment operator
    const SUNDIALSNVector& operator= (const SUNDIALSNVector& x)
    { *vector = *x.vector; return *this; }

  private:

    //--- Implementation of N_Vector ops

    // Get ID for custom SUNDIALSNVector implementation
    static N_Vector_ID N_VGetVectorID(N_Vector nv)
    {
      dolfin_debug("N_VGetVectorID");
      return SUNDIALS_NVEC_CUSTOM;
    }

    // Sets the components of the N_Vector z to be the absolute values of the
    // components of the N_Vector x
    static void N_VAbs(N_Vector x, N_Vector z)
    {
      dolfin_debug("N_VAbs");
      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vz = static_cast<SUNDIALSNVector *>(z->content)->vec();

      *vz = *vx;
      vz->abs();
    }

    /// Sets all components of the N_Vector z to realtype c.
    static void N_VConst(double c, N_Vector z)
    {
      dolfin_debug("N_VConst");
      auto v = static_cast<SUNDIALSNVector *>(z->content)->vec();
      *v = c;
    }

    /// Creates a new N_Vector of the same type as an existing vector z
    /// and sets the ops field.  It does not copy the vector, but rather
    /// allocates storage for the new vector.
    static N_Vector N_VClone(N_Vector z)
    {
      dolfin_debug("N_VClone");
      auto vz = static_cast<const SUNDIALSNVector *>(z->content);

      SUNDIALSNVector *new_vector = new SUNDIALSNVector(*vz);

      _generic_N_Vector *V = new _generic_N_Vector;
      V->ops = z->ops;
      V->content = (void *)(new_vector);

      return V;
    }

    /// Creates a new N_Vector of the same type as an existing vector
    /// x and sets the ops field. It does not allocate storage for data.
    static N_Vector N_VCloneEmpty(N_Vector x)
    {
      dolfin_debug("N_VCloneEmpty");
      dolfin_not_implemented();
      return NULL;
    }

    /// Destroys the N_Vector v and frees memory allocated for its
    /// internal data.
    static void N_VDestroy(N_Vector z)
    {
      dolfin_debug("N_VDestroy");
      delete (SUNDIALSNVector*)(z->content);
      delete z;
    }

    /// Returns storage requirements for one N_Vector. lrw contains the
    /// number of realtype words and liw contains the number of integer words.
    static void N_VSpace(N_Vector x, long int *y, long int *z)
    {
      dolfin_debug("N_VSpace");
      dolfin_not_implemented();
    }

    /// Returns a pointer to a realtype array from the N_Vector v.
    static double* N_VGetArrayPointer(N_Vector x)
    {
      dolfin_debug("N_VGetArrayPointer");
      dolfin_not_implemented();
      return NULL;
    }

    /// Overwrites the data in an N_Vector with a given array of realtype.
    static void N_VSetArrayPointer(double* c,N_Vector x)
    {
      dolfin_debug("N_VSetArrayPointer");
      dolfin_not_implemented();
    }

    /// Sets the N_Vector z to be the component-wise product of the N_Vector
    /// inputs x and y.
    static void N_VProd(N_Vector x, N_Vector y, N_Vector z)
    {
      dolfin_debug("N_VProd");
      auto vx = static_cast<const SUNDIALSNVector*>(x->content)->vec();
      auto vy = static_cast<const SUNDIALSNVector*>(y->content)->vec();
      auto vz = static_cast<SUNDIALSNVector*>(z->content)->vec();

      // Copy x to z
      *vz = *vx;
      // Multiply by y
      *vz *= *vy;
    }

    /// Sets the N_Vector z to be the component-wise ratio of the N_Vector
    /// inputs x and y
    static void N_VDiv(N_Vector x, N_Vector y, N_Vector z)
    {
      dolfin_debug("N_VDiv");

      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vy = static_cast<const SUNDIALSNVector *>(y->content)->vec();
      auto vz = static_cast<SUNDIALSNVector *>(z->content)->vec();

      std::vector<double> xdata;
      vx->get_local(xdata);
      std::vector<double> ydata;
      vy->get_local(ydata);
      for (unsigned int i = 0; i != xdata.size(); ++i)
        xdata[i] /= ydata[i];

      vz->set_local(xdata);
      vz->apply("insert");

    }

    /// Scales the N_Vector x by the double scalar c and returns the result in z
    static void N_VScale(double c, N_Vector x, N_Vector z)
    {
      dolfin_debug("N_VScale");
      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vz = static_cast<SUNDIALSNVector *>(z->content)->vec();

      // z = c*x
      *vz = *vx;
      *vz *= c;
    }

    /// Sets the components of the N_Vector z to be the inverses of the
    /// components of the N_Vector x
    static void N_VInv(N_Vector x, N_Vector z)
    {
      dolfin_debug("N_VInv");
      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vz = static_cast<SUNDIALSNVector *>(z->content)->vec();

      // z = 1/x
      std::vector<double> xvals;
      vx->get_local(xvals);
      for (auto &val : xvals)
        val = 1.0/val;
      vz->set_local(xvals);
      vz->apply("insert");
    }

    /// Adds the double scalar c to all components of x and returns the result
    /// in the N_Vector z
    static void N_VAddConst(N_Vector x, double c, N_Vector z)
    {
      dolfin_debug("N_VAddConst");
      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vz = static_cast<SUNDIALSNVector *>(z->content)->vec();

      *vz = *vx;
      *vz += c;
    }

    /// Returns the value of the ordinary dot product of x and y
    static double N_VDotProd(N_Vector x, N_Vector z)
    {
      dolfin_debug("N_VDotProd");
      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vz = static_cast<SUNDIALSNVector *>(z->content)->vec();

      return vx->inner(*vz);
    }

    /// Returns the maximum norm of the N_Vector x
    static double N_VMaxNorm(N_Vector x)
    {
      dolfin_debug("N_VMaxNorm");
      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vy = vx->copy();
      vy->abs();
      return vy->max();
    }

    /// Returns the smallest element of the N_Vector x
    static double N_VMin(N_Vector x)
    {
      dolfin_debug("N_VMin");
      return (static_cast<const SUNDIALSNVector *>(x->content)->vec())->min();
    }

    /// Performs the operation z = ax + by , where a and b are double scalars
    /// and x and y are of type N_Vector
    static void N_VLinearSum(double a, N_Vector x, double b, N_Vector y, N_Vector z)
    {
      dolfin_debug("N_VLinearSum");
      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vy = static_cast<const SUNDIALSNVector *>(y->content)->vec();
      auto vz = static_cast<SUNDIALSNVector *>(z->content)->vec();

      std::vector<double> xdata;
      vx->get_local(xdata);
      std::vector<double> ydata;
      vy->get_local(ydata);

      for (unsigned int i = 0; i != xdata.size(); ++i)
        xdata[i] = a*xdata[i] + b*ydata[i];

      vz->set_local(xdata);
      vz->apply("insert");
    }

    /// Returns  the  weighted  root-mean-square  norm  of  the N_Vector x with
    /// double weight vector w
    static double N_VWrmsNorm(N_Vector x, N_Vector z)
    {
      dolfin_debug("N_VWrmsNorm");
      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vz = static_cast<SUNDIALSNVector *>(z->content)->vec();

      auto y = vx->copy();
      *y *= *vz;
      return y->norm("l2")/std::sqrt(y->size());
    }

    /// Returns the weighted root mean square norm of the N_Vector x with double
    /// weight vector w built using only the elements of x corresponding to
    /// nonzero elements of the N_Vector id
    static double N_VWrmsNormMask(N_Vector x, N_Vector y, N_Vector z)
    {
      dolfin_debug("N_VWrmsNormMask");
      dolfin_not_implemented();
      return 0.0;
    }

    /// Returns the weighted Euclidean l2 norm  of the N_Vector x with double
    /// weight vector w
    static double N_VWl2Norm(N_Vector x, N_Vector z )
    {
      dolfin_debug("N_VWl2Norm");
      dolfin_not_implemented();
      return 0.0;
    }

    /// Returns the l1 norm of the N_Vector x
    static double N_VL1Norm(N_Vector x )
    {
      dolfin_debug("N_VL1Norm");
      dolfin_not_implemented();
      return 0.0;
    }

    /// Compares the components of the N_Vector x to the double scalar c and
    /// returns an N_Vector z
    static void N_VCompare(double c, N_Vector x, N_Vector z)
    {
      dolfin_debug("N_VCompare");
      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vz = static_cast<SUNDIALSNVector *>(z->content)->vec();
      std::vector<double> xvals;
      vx->get_local(xvals);
      for (auto &val : xvals)
        val = (std::abs(val) >= c) ? 1.0 : 0.0;
      vz->set_local(xvals);
      vz->apply("insert");
    }

    /// Sets the components of the N_Vector z to be the inverses of the
    /// components of the N_Vector x, with prior testing for zero values
    static int N_VInvTest(N_Vector x, N_Vector z)
    {
      dolfin_debug("N_VInvTest");
      int no_zero_found = true;
      auto vx = static_cast<const SUNDIALSNVector *>(x->content)->vec();
      auto vz = static_cast<SUNDIALSNVector *>(z->content)->vec();

      std::vector<double> xvals;
      vx->get_local(xvals);
      for (auto &val : xvals)
        if(val != 0)
	  val = 1.0/val;
	else
	  no_zero_found = false;
      vz->set_local(xvals);

      vz->apply("insert");

      return no_zero_found;
    }


    /// This routine returns the minimum of the quotients obtained by termwise
    /// dividing x by denom z
    static double N_VMinQuotient(N_Vector x, N_Vector z )
    {
      dolfin_debug("N_VConstrMask");
      dolfin_not_implemented();
      return 0.0;
    }

    /// Performs constraint tests
    static int N_VConstrMask(N_Vector x, N_Vector y, N_Vector z )
    {
      dolfin_debug("N_VConstrMask");
      dolfin_not_implemented();
      return 0;
    }

    // Pointer to concrete implementation
    std::shared_ptr<GenericVector> vector;

    // Pointer to SUNDIALS struct
    std::unique_ptr<_generic_N_Vector> N_V;

    // Structure containing function pointers to vector operations
    struct _generic_N_Vector_Ops ops = {N_VGetVectorID,        //   N_Vector_ID (*N_VGetVectorID)(SUNDIALSNVector);
                                        N_VClone,              //   NVector    (*N_VClone)(NVector);
                                        N_VCloneEmpty,         //   NVector    (*N_VCloneEmpty)(NVector);
                                        N_VDestroy,            //   void        (*N_VDestroy)(NVector);
                                        NULL,                  //N_VSpace,    //   void        (*N_VSpace)(NVector, long int *, long int *);
                                        N_VGetArrayPointer,    //   realtype*   (*N_VGetArrayPointer)(NVector);
                                        N_VSetArrayPointer,    //   void        (*N_VSetArrayPointer)(realtype *, NVector);
                                        N_VLinearSum,    //   void        (*N_VLinearSum)(realtype, NVector, realtype, NVector, NVector);
                                        N_VConst,          //   void        (*N_VConst)(realtype, NVector);
                                        N_VProd,    //   void        (*N_VProd)(NVector, NVector, NVector);
                                        N_VDiv,    //   void        (*N_VDiv)(NVector, NVector, NVector);
                                        N_VScale,    //   void        (*N_VScale)(realtype, NVector, NVector);
                                        N_VAbs,    //   void        (*N_VAbs)(NVector, NVector);
                                        N_VInv,    //   void        (*N_VInv)(NVector, NVector);
                                        N_VAddConst,    //   void        (*N_VAddConst)(NVector, realtype, NVector);
                                        N_VDotProd,    //   realtype    (*N_VDotProd)(NVector, NVector);
                                        N_VMaxNorm,    //   realtype    (*N_VMaxNorm)(NVector);
                                        N_VWrmsNorm,    //   realtype    (*N_VWrmsNorm)(NVector, NVector);
                                        N_VWrmsNormMask,    //   realtype    (*N_VWrmsNormMask)(NVector, NVector, NVector);
                                        N_VMin,    //   realtype    (*N_VMin)(NVector);
                                        N_VWl2Norm,    //   realtype    (*N_VWl2Norm)(NVector, NVector);
                                        N_VL1Norm,    //   realtype    (*N_VL1Norm)(NVector);
                                        N_VCompare,    //   void        (*N_VCompare)(realtype, NVector, NVector);
                                        N_VInvTest,    //   booleantype (*N_VInvtest)(NVector, NVector);
                                        N_VConstrMask,    //   booleantype (*N_VConstrMask)(NVector, NVector, NVector);
                                        N_VMinQuotient};    //   realtype    (*N_VMinQuotient)(NVector, NVector);
  };


}

#endif

#endif
