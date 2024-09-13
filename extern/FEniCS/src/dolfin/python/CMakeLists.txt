cmake_minimum_required(VERSION 3.5.0)

PROJECT(dolfin_pybind11)

# Configurable options for how we want to build
include(FeatureSummary)
option(DOLFIN_ENABLE_PETSC4PY "Compile with support for PETSc4py." ON)
option(DOLFIN_ENABLE_MPI4PY "Compile with support for MPI4py." ON)
add_feature_info(DOLFIN_ENABLE_PETSC4PY DOLFIN_ENABLE_PETSC4PY "Compile with support for PETSc4py.")
add_feature_info(DOLFIN_ENABLE_MPI4PY DOLFIN_ENABLE_MPI4PY "Compile with support for MPI4py.")

# Get options from env (difficult to pass them through setuptools)
if(DEFINED ENV{DOLFIN_ENABLE_PETSC4PY})
  set(DOLFIN_ENABLE_PETSC4PY $ENV{DOLFIN_ENABLE_PETSC4PY})
endif()
if(DEFINED ENV{DOLFIN_ENABLE_MPI4PY})
  set(DOLFIN_ENABLE_MPI4PY $ENV{DOLFIN_ENABLE_MPI4PY})
endif()

# Find required packages
find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# Strict compiler flags
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
#if (HAVE_PEDANTIC)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -pedantic")
#endif()

# Create the binding library
pybind11_add_module(cpp SHARED
  src/dolfin.cpp
  src/parameter.cpp
  src/adaptivity.cpp
  src/ale.cpp
  src/common.cpp
  src/fem.cpp
  src/function.cpp
  src/generation.cpp
  src/geometry.cpp
  src/graph.cpp
  src/log.cpp
  src/math.cpp
  src/mesh.cpp
  src/multistage.cpp
  src/ts.cpp
  src/io.cpp
  src/la.cpp
  src/nls.cpp
  src/refinement.cpp
  src/MPICommWrapper.cpp)

# Add DOLFIN libraries and other config
target_link_libraries(cpp PRIVATE pybind11::module dolfin)

# Add to CMake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Check for petsc4py
set(PETSC4PY_FOUND FALSE)
if(DOLFIN_ENABLE_PETSC4PY)
  find_package(PETSc4py)
endif()
if (PETSC4PY_FOUND)
  target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIRS})
  target_compile_definitions(cpp PRIVATE HAS_PYBIND11_PETSC4PY)
endif()

# Check for mpi4py
set(MPI4PY_FOUND FALSE)
if(DOLFIN_ENABLE_MPI4PY)
  find_package(MPI4PY)
endif()
if (MPI4PY_FOUND)
  target_include_directories(cpp PRIVATE ${MPI4PY_INCLUDE_DIR})
  target_compile_definitions(cpp PRIVATE HAS_PYBIND11_MPI4PY)
endif()
