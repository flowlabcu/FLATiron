# -*- coding: utf-8 -*-

import hashlib
import dijitso
import re
import sys

from dolfin.cpp.log import log, LogLevel
from . import get_pybind_include
import dolfin.cpp as cpp
from dolfin.jit.jit import dijitso_jit, dolfin_pc


def jit_generate(cpp_code, module_name, signature, parameters):

    log(LogLevel.TRACE, "Calling dijitso just-in-time (JIT) compiler for pybind11 code.")

    # Split code on reserved word "SIGNATURE" which will be replaced
    # by the module signature
    # This must occur only once in the code
    split_cpp_code = re.split('SIGNATURE', cpp_code)
    if len(split_cpp_code) < 2:
        raise RuntimeError("Cannot find keyword: SIGNATURE in pybind11 C++ code.")
    elif len(split_cpp_code) > 2:
        raise RuntimeError("Found multiple instances of keyword: SIGNATURE in pybind11 C++ code.")

    code_c = split_cpp_code[0] + signature + split_cpp_code[1]

    code_h = ""
    depends = []

    return code_h, code_c, depends


def compile_cpp_code(cpp_code, **kwargs):
    """Compile a user C(++) string and expose as a Python object with
    pybind11.

    Note: this is experimental

    """

    # Set compiler/build options
    # FIXME: need to locate Python libs and pybind11
    from distutils import sysconfig
    params = dijitso.params.default_params()
    pyversion = "python" + sysconfig.get_config_var("LDVERSION")
    params['cache']['lib_prefix'] = ""
    params['cache']['lib_basename'] = ""
    params['cache']['lib_loader'] = "import"

    extra_include_dirs = kwargs.get("include_dirs", [])
    extra_libraries = kwargs.get("libraries", [])
    extra_library_dirs = kwargs.get("library_dirs", [])
    cppargs = kwargs.get("cppargs", [])

    # Include path and library info from DOLFIN (dolfin.pc)
    params['build']['include_dirs'] = dolfin_pc["include_dirs"] + extra_include_dirs + get_pybind_include() + [sysconfig.get_config_var("INCLUDEDIR") + "/" + pyversion]
    params['build']['libs'] = dolfin_pc["libraries"] + extra_libraries
    params['build']['lib_dirs'] = dolfin_pc["library_dirs"] + extra_library_dirs + [sysconfig.get_config_var("LIBDIR")]

    params['build']['cxxflags'] += ('-fno-lto',)
    if sys.platform == 'darwin':
        # -undefined dynamic_lookup is needed because Python itself will resolve
        # symbols in libpython at runtime.
        # Linking libpython shouldn't be needed in general and doesn't work if Python
        # itself statically links libpython.
        # This should probably be in the default flags for dijitso.
        params['build']['cxxflags'] += ('-undefined', 'dynamic_lookup')

    for flag in cppargs:
        params['build']['cxxflags'] += (flag,)

    # Enable all macros from dolfin.pc
    dmacros = ()
    for dm in dolfin_pc['define_macros']:
        if dm[1] is None or len(dm[1]) == 0:
            dmacros += ('-D' + dm[0],)
        else:
            dmacros += ('-D' + dm[0] + '=' + dm[1],)

    params['build']['cxxflags'] += dmacros

    # This seems to be needed by OSX but not in Linux
    # FIXME: probably needed for other libraries too
    # if cpp.common.has_petsc():
    #     import os
    #     params['build']['libs'] += ['petsc']
    #     params['build']['lib_dirs'] += [os.environ["PETSC_DIR"] + "/lib"]

    hash_str = cpp_code + cpp.__version__
    module_hash = hashlib.md5(hash_str.encode('utf-8')).hexdigest()
    module_name = "dolfin_cpp_module_" + module_hash

    module, signature = dijitso_jit(cpp_code, module_name, params,
                                    generate=jit_generate)

    return module
