Changelog
=========

2019.1.0 (2019-04-19)
---------------------

- No changes.

2018.1.0 (2018-06-14)
---------------------

- Very minor cleanups

2017.2.0 (2017-12-05)
---------------------

- Add support for linking external libraries
- Add support for creating a loadable module with __import__ with "lib_loader" parameter (ctypes, module)

2017.1.0.post1 (2017-09-12)
---------------------

- Change PyPI package name to fenics-dijitso.

2017.1.0 (2017-05-09)
---------------------

- Minor fixes

2016.2.0 (2016-11-30)
---------------------

- Introduce commandline script ``dijitso`` with various subcommands to
  interact with the cache
- Improve extraction of source files to reproduce compilation failure
  during jit
- Implement support for linking between jit modules
- Add optional dependency on ``subprocess32`` to handle fork safety on
  infiniband clusters
- Remove ``instant`` dependency

2016.1.0 (2016-06-23)
---------------------

- Initial implementation
