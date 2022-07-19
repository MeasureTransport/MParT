Installation
------------

Install from Conda
==================

.. panels::
    :container: container-lg pb-3
    :column: col-lg-12 p-2

    ^^^^^^^^^^^^^^^^^^^
    COMING SOON!


    ++++++++++++++++++++++

    .. code-block:: bash

        conda install -c conda-forge mpart

    ---
    :column: col-lg-12 p-2

.. _installation:


Compiling from Source
=====================
MParT uses CMake to handle dependencies and compiler configurations.   A basic build of MParT that should work on most operating systems can be obtained with:

.. code-block:: bash

   mkdir build
   cd build
   cmake                                         \
     -DCMAKE_INSTALL_PREFIX=<your/install/path>  \
     -DKokkos_ENABLE_PTHREAD=ON                  \
     -DKokkos_ENABLE_SERIAL=ON                   \
   ..
   make install

This will compile the :code:`mpart` library and the python bindings. If you are compiling on a multicore machine, you can use :code:`make -j N_JOBS install`, where :code:`N_JOBS` is the number of processes the computer can compile with in parallel.  This installation should also automatically install and build Kokkos, Eigen, and Catch2, assuming they aren't installed already. If CMake has trouble finding prior installations of these, then you can configuring CMake using:

.. code-block:: bash

    cmake                                        \
     -DCMAKE_INSTALL_PREFIX=<your/install/path>  \
     -DKokkos_ROOT=<your/kokkos/install/root>    \
     -DEigen3_ROOT=<your/eigen3/install/root>    \
     -DCatch2_ROOT=<your/catch2/install/root>    \
     -DKokkos_ENABLE_PTHREAD=ON                  \
     -DKokkos_ENABLE_SERIAL=ON                   \
   ..

Feel free to mix and match previous installations of Eigen, Kokkos, Pybind11, and Catch2 with submodules you don't already have using these :code:`X_ROOT` flags. Note that Catch2 and Kokkos in this example will need to be compiled with shared libraries. MParT has not been tested with all versions of all dependencies, but it does require CMake version >=3.13. Further, it has been tested with Kokkos 3.6.0, Eigen 3.4.0, Pybind11 2.9.2, and Catch2 3.0.0-preview3 (there are some issues encountered when compiling MParT with Catch2 3.0.1).

The command :code:`make install` will also create a test executable called :code:`RunTests` in the :code:`build` directory.  The tests can be run with:

.. code-block::

   ./RunTests

Or, with the additional specification of the number of Kokkos threads to use:

.. code-block::

   ./RunTests --kokkos-threads=4


.. tip::
   Depending on your python configuration, pybind11 may throw an error during configuration that looks like

   .. code-block::

      CMake Error in bindings/python/CMakeLists.txt:
        Imported target "pybind11::module" includes non-existent path

   This often results when due to conda environment mismatches, but can typically be circumvented by explicitly setting the path to your python executable.  When calling cmake, add :code:`-DPYTHON_EXECUTABLE=`which python``.

.. tip:: 
  On OSX, using MParT with the system version of python might result in an error with something like:
  
  .. code-block::

    ImportError: dlopen(pympart.so, 2): no suitable image found.  Did find:
        MParT/python/mpart/pympart.so: mach-o, but wrong architecture
        MParT/python/mpart/pympart.so: mach-o, but wrong architecture

  You can sometimes force OSX to use the x86_64 version of python using the :code:`arch` executable.   For example, to run a script :code:`test.py`, you can use 

  .. code-block::

    arch -x86_64 /usr/bin/python test.py

Compiling with Julia Bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, MParT will look for Julia during configuration and will attempt to build the Julia bindings if the Julia :code:`CxxWrap` package is installed.   To install :code:`CxxWrap`, run the following command in your Julia prompt:

.. code-block:: julia

    import Pkg; Pkg.add("CxxWrap")

If you have Julia installed, but CMake was not able to find it during MParT configuration, you may need to manually specify :code:`JULIA_EXE` variable during configuration.  For example, adding :code:`-DJULIA_EXE=~/opt/anaconda3/envs/mpart/bin/julia` will tell CMake to use the Julia executable installed by anaconda in the :code:`mpart` conda environment.

To prevent the Julia bindings from being compiled, even if Julia and CxxWrap are found, set :code:`MPART_JULIA=OFF` during the CMake configuration.

Compiling with CUDA Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To support a GPU at the moment, you need a few special requirements. Due to the way that Kokkos handles GPU code, MParT must be compiled using a special wrapper around NVCC that Kokkos provides. First, we compile Kokkos with the required options:

.. code-block:: bash

    cmake \
        -DCMAKE_INSTALL_PREFIX=</new/kokkos/install/path> \
        -DBUILD_SHARED_LIBS=ON                            \
        -DKokkos_ENABLE_SERIAL=OFF                        \
        -DKokkos_ENABLE_OPENMP=ON                         \
        -DKokkos_ENABLE_CUDA=ON                           \
        -DKokkos_ARCH_VOLTA70=ON                          \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON                    \
        -DKokkos_CUDA_DIR=<cuda/install/path>             \
        -DKokkos_CXX_STANDARD=17                          \
    ../

Replace the :code:`Kokkos_ARCH_VOLTA70` as needed with whatever other arch the compute resource uses that Kokkos supports. If you aren't sure, try omitting this as Kokkos has some machinery to detect such architecture.

.. tip::
    If you're getting an error about C++ standards, try using a new version of your compiler; :code:`g++`, for example, does not support the flag :code:`--std=c++17` below version 8, where :code:`nvcc` only supports such syntax. For more details, see `this issue <https://github.com/kokkos/kokkos/issues/5157>`_ in Kokkos.

Using the above documentation on building with an external install of Kokkos, we can then configure MParT once in the :code:`build` directory using the following command:

.. code-block:: bash

    cmake \
        -DCMAKE_INSTALL_PREFIX=<your/install/path>                       \
        -DKokkos_ROOT=</new/kokkos/install/path>                         \
        -DCMAKE_CXX_COMPILER=</new/kokkos/install/path>/bin/nvcc_wrapper \
    ..

Make sure that :code:`CMAKE_CXX_COMPILER` uses a full path from the root!

.. tip::
   If you're using a Power8 or Power9 architecture, Eigen may give you trouble when trying to incorporate vectorization using Altivec, specifically when compiling for GPU. In this case, go into :code:`CMakeFiles.txt` and add :code:`add_compile_definition(EIGEN_DONT_VECTORIZE)`.


