# FEniCS Legacy build scripts from source:w


This is a from source install instructions for FEniCS Legacy. The user must have mpi compiler available. To build FEniCS, navigate to the `build`file, and edit the `MPI_DIR`variable to the directory containing `bin/mpicc`.

Once the mpi compiler directory is provided, run the `build`script:

```bash
./build
```
This script will create build FEniCS Legacy and all of its dependencies. This will take a little bit of time. Once compilation is complete, this script creates a file called `fenics_env`which contains the path to the installations. Simply run
```bash
source fenics_env
```
to set the correct environment variables.

