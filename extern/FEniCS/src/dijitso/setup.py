# -*- coding: utf-8 -*-
from setuptools import setup
from os.path import join
import sys

module_name = "dijitso"

if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)

# Set version
version = "2019.1.0"

url = "https://bitbucket.org/fenics-project/%s/" % module_name
tarball = None
if 'dev' not in version:
    tarball = url + "downloads/%s-%s.tar.gz" % (module_name, version)

script_names = ("dijitso",)
entry_points = {'console_scripts': ['dijitso = dijitso.__main__:main']}

scripts = [join("scripts", script) for script in script_names]
man_files = [join("doc", "man", "man1", "%s.1.gz" % (script,)) for script in script_names]
data_files = [(join("share", "man", "man1"), man_files)]


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Operating System :: POSIX
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering :: Mathematics
Topic :: Software Development :: Libraries :: Python Modules
"""

requires = ["numpy"]

setup(name="fenics-dijitso",
      version=version,
      description="Distributed just-in-time building of shared libraries",
      author="Martin Sandve Alnæs",
      author_email="martinal@simula.no",
      url=url,
      download_url=tarball,
      classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
      # scripts=scripts,  # Using entry_points instead
      entry_points=entry_points,
      packages=["dijitso"],
      package_dir={'dijitso': 'dijitso'},
      install_requires=requires,
      data_files=data_files
      )
