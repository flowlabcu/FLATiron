# - Try to find SUNDIALS
# Once done this will define
#
#  SUNDIALS_FOUND        - system has SUNDIALS
#  SUNDIALS_INCLUDE_DIRS - include directories for SUNDIALS
#  SUNDIALS_LIBRARIES    - libraries for SUNDIALS
#  SUNDIALS_VERSION      - version for SUNDIALS

#=============================================================================
# Copyright (C) 2010 Garth N. Wells, Anders Logg and Johannes Ring
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

if (MPI_CXX_FOUND)
  find_path(SUNDIALS_INCLUDE_DIRS
    NAMES cvode/cvode.h
    HINTS ${SUNDIALS_DIR}/include $ENV{SUNDIALS_DIR}/include ${PETSC_INCLUDE_DIRS}
    DOC "Directory where the SUNDIALS CVODE header files are located"
  )
  find_library(SUNDIALS_LIBRARY 
    NAMES sundials_cvode
    HINTS ${SUNDIALS_DIR}/lib $ENV{SUNDIALS_DIR}/lib ${PETSC_LIBRARY_DIRS}
    NO_DEFAULT_PATH
    DOC "Directory where the SUNDIALS CVODE library is located"
  )
  find_library(SUNDIALS_NVECTOR_SERIAL_LIBRARY
    NAMES sundials_nvecserial
    HINTS ${SUNDIALS_DIR}/lib $ENV{SUNDIALS_DIR}/lib ${PETSC_LIBRARY_DIRS}
    NO_DEFAULT_PATH
    DOC "Directory where the SUNDIALS CVODE library is located"
  )

  set(SUNDIALS_LIBRARIES ${SUNDIALS_LIBRARY})
  set(SUNDIALS_LIBRARIES ${SUNDIALS_LIBRARIES} ${SUNDIALS_NVECTOR_PARALLEL_LIBRARY})
  set(SUNDIALS_LIBRARIES ${SUNDIALS_LIBRARIES} ${SUNDIALS_NVECTOR_SERIAL_LIBRARY})

  # Try compiling and running test program
  if (DOLFIN_SKIP_BUILD_TESTS)
    set(SUNDIALS_TEST_RUNS TRUE)
    set(SUNDIALS_VERSION "UNKNOWN")
    set(SUNDIALS_VERSION_OK TRUE)
  elseif (SUNDIALS_INCLUDE_DIRS AND SUNDIALS_LIBRARIES)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES  ${CMAKE_REQUIRED_INCLUDES} ${MPI_CXX_INCLUDE_PATH})
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES}    ${MPI_CXX_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS     ${CMAKE_REQUIRED_FLAGS}  ${MPI_CXX_COMPILE_FLAGS})

  set(CMAKE_REQUIRED_INCLUDES  ${CMAKE_REQUIRED_INCLUDES} ${SUNDIALS_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${SUNDIALS_LIBRARIES})

  # Check SUNDIALS version
  set(SUNDIALS_CONFIG_TEST_VERSION_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/sundials_config_test_version.cpp")
  file(WRITE ${SUNDIALS_CONFIG_TEST_VERSION_CPP} "
#define MPICH_IGNORE_CXX_SEEK 1
#include <iostream>
#include \"cvode/cvode.h\"

int main() {
#ifdef SUNDIALS_VERSION
  std::cout << SUNDIALS_VERSION;
#elif defined(SUNDIALS_PACKAGE_VERSION)
  std::cout << SUNDIALS_PACKAGE_VERSION;
#else
  std::cout << SUNDIALS_VERSION_MAJOR << \".\"
  << SUNDIALS_VERSION_MINOR;
#endif
  return 0;
}
")

    try_run(
      SUNDIALS_CONFIG_TEST_VERSION_EXITCODE
      SUNDIALS_CONFIG_TEST_VERSION_COMPILED
      ${CMAKE_CURRENT_BINARY_DIR}
      ${SUNDIALS_CONFIG_TEST_VERSION_CPP}
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
      	"-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
      COMPILE_OUTPUT_VARIABLE SUNDIALS_CONFIG_TEST_VERSION_COMPILE_OUTPUT
      RUN_OUTPUT_VARIABLE SUNDIALS_CONFIG_TEST_VERSION_OUTPUT
      )

    if (SUNDIALS_CONFIG_TEST_VERSION_EXITCODE EQUAL 0)
      set(SUNDIALS_VERSION ${SUNDIALS_CONFIG_TEST_VERSION_OUTPUT} CACHE TYPE STRING)
      mark_as_advanced(SUNDIALS_VERSION)
    endif()

    if (SUNDIALS_FIND_VERSION)
      string(REPLACE "." ";" VERSION_LIST ${SUNDIALS_VERSION})
      list(GET VERSION_LIST 0 SUNDIALS_VERSION_MAJOR)
      list(GET VERSION_LIST 1 SUNDIALS_VERSION_MINOR)
      # Check if version found is >= required version
      if (NOT "${SUNDIALS_VERSION_MAJOR}" VERSION_LESS "3")
      	set(SUNDIALS_VERSION_OK TRUE)
      endif()
    else()
      # No specific version requested
      set(SUNDIALS_VERSION_OK TRUE)
    endif()
    mark_as_advanced(SUNDIALS_VERSION_OK)

    # Build and run test program
    set(SUNDIALS_TEST_LIB_CPP "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/sundials_test_lib.cpp")
    file(WRITE ${SUNDIALS_TEST_LIB_CPP} "
#define MPICH_IGNORE_CXX_SEEK 1
#define NEQ 10
#include <iostream>
#include <nvector/nvector_serial.h>
#include <cvode/cvode.h>

int main(int argc, char** argv)
{
  int my_pe, npes;
  long int local_N, nperpe, nrem;
  N_Vector u;  

  u = N_VNew_Serial(NEQ);
  N_VDestroy_Serial(u);

  return 0;
}") 



    message(STATUS "Performing test SUNDIALS_TEST_RUNS")
    try_run(
      SUNDIALS_TEST_LIB_EXITCODE
      SUNDIALS_TEST_LIB_COMPILED
      ${CMAKE_CURRENT_BINARY_DIR}
      ${SUNDIALS_TEST_LIB_CPP}
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
        "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
      COMPILE_OUTPUT_VARIABLE SUNDIALS_TEST_LIB_COMPILE_OUTPUT
      RUN_OUTPUT_VARIABLE SUNDIALS_TEST_LIB_OUTPUT
      )

    if (SUNDIALS_TEST_LIB_COMPILED AND SUNDIALS_TEST_LIB_EXITCODE EQUAL 0)
      message(STATUS "Performing test SUNDIALS_TEST_RUNS - Success")
      set(SUNDIALS_TEST_RUNS TRUE)
    else()
      message(STATUS "Performing test SUNDIALS_TEST_RUNS - Failed")
      if (SUNDIALS_DEBUG)
        # Output some variables
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "SUNDIALS_TEST_LIB_COMPILED = ${SUNDIALS_TEST_LIB_COMPILED}")
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "SUNDIALS_TEST_LIB_COMPILE_OUTPUT = ${SUNDIALS_TEST_LIB_COMPILE_OUTPUT}")
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "SUNDIALS_TEST_LIB_EXITCODE = ${SUNDIALS_TEST_LIB_EXITCODE}")
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "SUNDIALS_TEST_LIB_OUTPUT = ${SUNDIALS_TEST_LIB_OUTPUT}")
      endif()
    endif()
  endif()
endif()
# Standard package handling
find_package_handle_standard_args(SUNDIALS
                                  "SUNDIALS could not be found/configured."
                                  SUNDIALS_LIBRARIES
				  SUNDIALS_TEST_RUNS
                                  SUNDIALS_INCLUDE_DIRS
				  SUNDIALS_VERSION
				  SUNDIALS_VERSION_OK)
